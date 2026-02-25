"""
Model Version Registry
======================
Tracks every trained version of the AML model suite with its status,
metrics, training checkpoint, and on-disk artifacts.

Version lifecycle
-----------------
  full-retrain  →  creates / replaces baseline  (status = "baseline")
  incremental   →  creates experimental          (status = "experimental")
  auto/manual   →  promotes experimental         (status = "baseline")
  superseded    →  old baseline becomes          (status = "retired")

On-disk layout
--------------
  models_saved/
    versions.json            — manifest (single source of truth)
    versions/
      v1/                    — archived artifacts for v1
        aml_model.joblib
        aml_scaler.joblib
        svm_model.joblib
        anomaly_index.faiss
        anomaly_meta.pkl
        graphsage_model.npz
        graphsage_meta.pkl
        version_meta.json    — same data as manifest entry, for convenience
      v2/
        ...
  (Active baseline files stay in models_saved/ root — no changes to
   existing loading code.)
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_MODELS_DIR    = Path("models_saved")
_VERSIONS_DIR  = _MODELS_DIR / "versions"
_MANIFEST_PATH = _MODELS_DIR / "versions.json"

# Files that constitute a complete model snapshot
_ARTIFACT_FILES = [
    "aml_model.joblib",
    "aml_scaler.joblib",
    "svm_model.joblib",
    "svm_model_fp32.onnx",
    "xgb_model_fp32.onnx",
    "anomaly_index.faiss",
    "anomaly_meta.pkl",
    "graphsage_model.npz",
    "graphsage_meta.pkl",
    "training_metadata.json",
]


@dataclass
class ModelVersion:
    version_id: str                    # "v1", "v2", …
    status: str                        # "baseline" | "experimental" | "retired"
    trained_at: str                    # ISO-8601 UTC timestamp
    n_samples: int                     # training rows used
    fraud_rate: float                  # fraud prevalence in training set
    last_txn_timestamp: str            # latest transaction ts seen during training
    training_type: str = "full"        # "full" | "incremental"
    trigger: str = "manual"            # "manual" | "auto" | "schedule"
    metrics: dict = field(default_factory=dict)   # per-model roc_auc / avg_precision
    promotion_reason: str = ""         # set when promoted
    notes: str = ""

    # ── Convenience helpers ────────────────────────────────────────────────────

    def xgb_auc(self) -> float:
        return float(self.metrics.get("xgb", {}).get("roc_auc", 0.0))

    def svm_auc(self) -> float:
        return float(self.metrics.get("svm", {}).get("roc_auc", 0.0))

    def is_better_than(self, other: "ModelVersion", threshold: float = 0.99) -> bool:
        """Return True when this version's XGBoost AUC >= other's AUC * threshold."""
        return self.xgb_auc() >= other.xgb_auc() * threshold

    @property
    def artifact_dir(self) -> Path:
        return _VERSIONS_DIR / self.version_id


class VersionRegistry:
    """
    JSON-backed registry of all model versions.
    Thread-safe for reads; writes should be called from a single background
    thread (training pipeline).
    """

    def __init__(self):
        self._manifest = self._load_manifest()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        if _MANIFEST_PATH.exists():
            with open(_MANIFEST_PATH) as f:
                return json.load(f)
        return {"baseline": None, "experimental": None, "versions": {}}

    def _save_manifest(self) -> None:
        _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_MANIFEST_PATH, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)

    def reload(self) -> None:
        """Reload from disk (useful in long-running API process)."""
        self._manifest = self._load_manifest()

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_baseline(self) -> Optional[ModelVersion]:
        vid = self._manifest.get("baseline")
        if vid and vid in self._manifest["versions"]:
            return ModelVersion(**self._manifest["versions"][vid])
        return None

    def get_experimental(self) -> Optional[ModelVersion]:
        vid = self._manifest.get("experimental")
        if vid and vid in self._manifest["versions"]:
            return ModelVersion(**self._manifest["versions"][vid])
        return None

    def list_versions(self) -> list[ModelVersion]:
        return [
            ModelVersion(**v)
            for v in self._manifest["versions"].values()
        ]

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        d = self._manifest["versions"].get(version_id)
        return ModelVersion(**d) if d else None

    def next_version_id(self) -> str:
        nums = []
        for vid in self._manifest.get("versions", {}):
            try:
                nums.append(int(vid[1:]))   # "v3" → 3
            except (ValueError, IndexError):
                pass
        return f"v{max(nums, default=0) + 1}"

    # ── Mutations ──────────────────────────────────────────────────────────────

    def register_version(self, version: ModelVersion) -> None:
        self._manifest["versions"][version.version_id] = asdict(version)
        # Also write a per-version meta file for easy inspection
        version.artifact_dir.mkdir(parents=True, exist_ok=True)
        meta_path = version.artifact_dir / "version_meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(version), f, indent=2, default=str)
        self._save_manifest()

    def set_baseline(self, version_id: str,
                     reason: str = "manually promoted") -> None:
        """Promote version_id to baseline; retire the previous one."""
        old_id = self._manifest.get("baseline")
        if old_id and old_id != version_id and old_id in self._manifest["versions"]:
            self._manifest["versions"][old_id]["status"] = "retired"

        self._manifest["baseline"] = version_id
        if version_id in self._manifest["versions"]:
            self._manifest["versions"][version_id]["status"] = "baseline"
            self._manifest["versions"][version_id]["promotion_reason"] = reason

        # Clear experimental pointer if this version was the experimental one
        if self._manifest.get("experimental") == version_id:
            self._manifest["experimental"] = None

        self._save_manifest()

    def set_experimental(self, version_id: str) -> None:
        # Retire previous experimental if different
        old_exp = self._manifest.get("experimental")
        if old_exp and old_exp != version_id and old_exp in self._manifest["versions"]:
            self._manifest["versions"][old_exp]["status"] = "retired"

        self._manifest["experimental"] = version_id
        if version_id in self._manifest["versions"]:
            self._manifest["versions"][version_id]["status"] = "experimental"
        self._save_manifest()

    def retire_version(self, version_id: str) -> None:
        if version_id in self._manifest["versions"]:
            self._manifest["versions"][version_id]["status"] = "retired"
        if self._manifest.get("experimental") == version_id:
            self._manifest["experimental"] = None
        self._save_manifest()

    # ── Artifact management ────────────────────────────────────────────────────

    def archive_current_models(self, version_id: str) -> Path:
        """
        Copy active model files from models_saved/ → models_saved/versions/{version_id}/.
        Returns the destination directory path.
        """
        dest = _VERSIONS_DIR / version_id
        dest.mkdir(parents=True, exist_ok=True)
        for fname in _ARTIFACT_FILES:
            src = _MODELS_DIR / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
        return dest

    def restore_version(self, version_id: str) -> bool:
        """
        Copy archived artifacts for version_id back to models_saved/ root
        (making that version the active one on disk).
        Returns True if successful.
        """
        src_dir = _VERSIONS_DIR / version_id
        if not src_dir.exists():
            return False
        for fname in _ARTIFACT_FILES:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, _MODELS_DIR / fname)
        return True


# ── Module-level singleton ─────────────────────────────────────────────────────

_registry: Optional[VersionRegistry] = None


def get_version_registry() -> VersionRegistry:
    global _registry
    if _registry is None:
        _registry = VersionRegistry()
    return _registry


def reset_version_registry() -> None:
    global _registry
    _registry = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
