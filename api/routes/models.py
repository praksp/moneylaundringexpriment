"""
Model Version Management API
=============================
Endpoints for managing model versions, triggering incremental training,
and promoting experimental models to baseline.

GET  /models/versions                → list all versions (newest first)
GET  /models/versions/current        → baseline + experimental + comparison
POST /models/train/incremental       → trigger incremental training (background)
GET  /models/train/status            → status of latest training job
POST /models/versions/{vid}/promote  → manually promote version to baseline
POST /models/versions/{vid}/retire   → retire a version
GET  /models/versions/{vid}/compare  → compare version vs baseline metrics
"""
import asyncio
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from auth.dependencies import require_admin
from ml.version import (
    ModelVersion, get_version_registry, reset_version_registry, now_iso,
)

router = APIRouter(prefix="/models", tags=["Model Versions"])

# ── In-process training job tracker ───────────────────────────────────────────

class _JobState:
    def __init__(self):
        self.running = False
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

_job = _JobState()
_job_lock = threading.Lock()


def _run_incremental_job(trigger: str, force: bool, auto_promote: bool) -> None:
    global _job
    try:
        from ml.incremental import run_incremental
        result = run_incremental(trigger=trigger, force=force,
                                 auto_promote=auto_promote)
        with _job_lock:
            _job.result       = result
            _job.finished_at  = now_iso()
            _job.running      = False
    except Exception as e:
        with _job_lock:
            _job.error       = str(e)
            _job.finished_at = now_iso()
            _job.running     = False


# ── Request / response models ──────────────────────────────────────────────────

class IncrementalTrainRequest(BaseModel):
    trigger:      str  = "manual"
    force:        bool = False      # train even if < MIN_SAMPLES_TO_PROMOTE
    auto_promote: bool = True       # auto-promote if accuracy meets threshold


class PromoteRequest(BaseModel):
    reason: str = "Manually promoted via API"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _version_to_dict(v: ModelVersion) -> dict:
    d = asdict(v)
    # Flatten top-level xgb / svm AUC for easy frontend consumption
    d["xgb_auc"] = v.xgb_auc()
    d["svm_auc"] = v.svm_auc()
    return d


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/versions", dependencies=[Depends(require_admin)])
def list_versions():
    """List all model versions ordered newest → oldest."""
    reg = get_version_registry()
    reg.reload()
    versions = sorted(
        reg.list_versions(),
        key=lambda v: v.trained_at,
        reverse=True,
    )
    return {
        "versions":       [_version_to_dict(v) for v in versions],
        "baseline_id":    reg._manifest.get("baseline"),
        "experimental_id":reg._manifest.get("experimental"),
        "total":          len(versions),
    }


@router.get("/versions/current", dependencies=[Depends(require_admin)])
def get_current_versions():
    """Return the active baseline and experimental versions with comparison."""
    reg = get_version_registry()
    reg.reload()
    baseline     = reg.get_baseline()
    experimental = reg.get_experimental()

    comparison = None
    if baseline and experimental:
        b_auc = baseline.xgb_auc()
        e_auc = experimental.xgb_auc()
        comparison = {
            "xgb_auc_delta":     round(e_auc - b_auc, 4),
            "would_auto_promote": experimental.is_better_than(baseline),
            "promotion_threshold": 0.99,
        }

    return {
        "baseline":     _version_to_dict(baseline)     if baseline     else None,
        "experimental": _version_to_dict(experimental) if experimental else None,
        "comparison":   comparison,
    }


@router.get("/versions/{version_id}", dependencies=[Depends(require_admin)])
def get_version(version_id: str):
    """Return full details for a specific version."""
    reg = get_version_registry()
    reg.reload()
    v = reg.get_version(version_id)
    if v is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    return _version_to_dict(v)


@router.get("/versions/{version_id}/compare", dependencies=[Depends(require_admin)])
def compare_version(version_id: str):
    """Side-by-side comparison of version_id vs the current baseline."""
    reg = get_version_registry()
    reg.reload()
    version  = reg.get_version(version_id)
    baseline = reg.get_baseline()

    if version is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")

    if baseline is None:
        raise HTTPException(status_code=404, detail="No baseline version found")

    def _diff(key: str) -> Optional[float]:
        v_val = version.metrics.get("xgb", {}).get(key)
        b_val = baseline.metrics.get("xgb", {}).get(key)
        if v_val is not None and b_val is not None:
            return round(float(v_val) - float(b_val), 4)
        return None

    return {
        "version":      _version_to_dict(version),
        "baseline":     _version_to_dict(baseline),
        "xgb_auc_delta": _diff("roc_auc"),
        "svm_auc_delta": None,
        "would_promote": version.is_better_than(baseline),
        "n_samples_delta": version.n_samples - baseline.n_samples,
    }


@router.post("/train/incremental", dependencies=[Depends(require_admin)])
async def trigger_incremental_training(req: IncrementalTrainRequest):
    """
    Start incremental training in the background.
    Loads only transactions newer than the baseline checkpoint, fine-tunes
    all models, and creates an experimental version (auto-promotes if good).
    """
    global _job
    with _job_lock:
        if _job.running:
            return {
                "status":  "already_running",
                "started_at": _job.started_at,
                "message": "An incremental training job is already in progress.",
            }
        _job.running     = True
        _job.started_at  = now_iso()
        _job.finished_at = None
        _job.result      = None
        _job.error       = None

    t = threading.Thread(
        target=_run_incremental_job,
        args=(req.trigger, req.force, req.auto_promote),
        daemon=True,
    )
    t.start()

    return {
        "status":     "started",
        "started_at": _job.started_at,
        "message":    "Incremental training started in background.",
        "poll_url":   "/models/train/status",
    }


@router.get("/train/status")
def training_status():
    """Poll the status of the latest incremental training job."""
    with _job_lock:
        return {
            "running":     _job.running,
            "started_at":  _job.started_at,
            "finished_at": _job.finished_at,
            "result":      _job.result,
            "error":       _job.error,
        }


@router.post("/versions/{version_id}/promote", dependencies=[Depends(require_admin)])
def promote_version(version_id: str, req: PromoteRequest):
    """
    Manually promote an experimental or any version to baseline.
    Reloads the ML registry so new models are live immediately.
    """
    reg = get_version_registry()
    reg.reload()
    v = reg.get_version(version_id)
    if v is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")

    # Restore the version's artifacts to models_saved/ root
    restored = reg.restore_version(version_id)
    if not restored:
        raise HTTPException(
            status_code=400,
            detail=f"Artifact directory not found for {version_id}. Cannot restore.",
        )

    reg.set_baseline(version_id, reason=req.reason)
    reset_version_registry()

    # Reload ML singletons with the restored models
    from ml.model import reset_registry
    from ml.anomaly import reset_detector
    reset_registry()
    reset_detector()

    return {
        "status":     "promoted",
        "version_id": version_id,
        "reason":     req.reason,
        "message":    f"{version_id} is now the active baseline.",
    }


@router.post("/versions/{version_id}/retire", dependencies=[Depends(require_admin)])
def retire_version(version_id: str):
    """Retire a version (keeps artifacts but removes it from active rotation)."""
    reg = get_version_registry()
    reg.reload()
    v = reg.get_version(version_id)
    if v is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")

    baseline = reg.get_baseline()
    if baseline and baseline.version_id == version_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot retire the active baseline. Promote another version first.",
        )

    reg.retire_version(version_id)
    reset_version_registry()
    return {"status": "retired", "version_id": version_id}
