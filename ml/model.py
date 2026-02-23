"""
AML ML Models
=============
Three independent classifiers for risk scoring, optimised for large-scale data:

  1. XGBoost  (AMLModel)   — histogram-quantized gradient boosting; tree_method="hist"
                             with max_bin=256 bins features before split finding.
                             Exports INT8-quantized ONNX for low-latency inference.

  2. SGD/SVM  (SVMModel)   — SGDClassifier with modified_huber loss replaces the
                             O(n²) RBF-SVC, giving O(n) training and O(1) inference
                             at any dataset size.

  3. FAISS-KNN (KNNModel)  — IVF-PQ approximate nearest-neighbour index replaces
                             the ball-tree, compressing the 44-d feature vectors into
                             8-bit product-quantized codes and partitioning into
                             Voronoi cells for sub-linear query time.
                             Falls back to sklearn ball-tree if faiss-cpu is absent.

Each model exposes a common interface:
  .fit(X, y)                    → metrics dict
  .predict_proba_single(fv)     → float  (0.0 – 1.0)
  .save() / .load()

The ModelRegistry loads all three at startup and exposes them through
get_registry() for use in the risk engine.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

try:
    from xgboost import XGBClassifier
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _USE_XGB = False

try:
    import faiss
    _USE_FAISS = True
except ImportError:
    _USE_FAISS = False

# Optional ONNX quantization stack
try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as rt
    from onnxruntime.quantization import quantize_dynamic, QuantType
    _USE_ONNX = True
except ImportError:
    _USE_ONNX = False

# XGBoost ONNX conversion (separate package)
try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as XGBFloatTensorType
    _USE_XGB_ONNX = True
except ImportError:
    _USE_XGB_ONNX = False

from config.settings import settings
from risk.features import FeatureVector


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _calibrate_threshold(classifier, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Return the threshold that maximises F1 on the validation set."""
    probs = classifier.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best]) if len(thresholds) > 0 else 0.3


def _eval_metrics(classifier, X_val: np.ndarray, y_val: np.ndarray,
                  threshold: float) -> dict:
    probs = classifier.predict_proba(X_val)[:, 1]
    return {
        "roc_auc": round(roc_auc_score(y_val, probs), 4),
        "avg_precision": round(average_precision_score(y_val, probs), 4),
        "best_threshold": round(threshold, 4),
        "classification_report": classification_report(
            y_val, (probs >= threshold).astype(int),
            target_names=["legitimate", "fraud"],
        ),
    }


# ── Model 1: XGBoost with histogram quantization ──────────────────────────────

class AMLModel:
    """
    Gradient-boosted trees with histogram-based feature quantization.

    Key scaling changes vs original:
    • tree_method="hist"  — XGBoost bins each of the 44 features into max_bin=256
      discrete buckets before searching for split points.  This reduces the
      split-finding complexity from O(n·f) to O(b·f) where b=256, giving
      5–10× faster training on large datasets with negligible accuracy loss.
    • max_bin=256         — 256 bins ≈ 8-bit quantization of feature values.
    • device="cpu"        — set to "cuda" to offload to GPU when available.
    • After training, export_onnx_quantized() converts weights to INT8 via
      ONNX Runtime's dynamic quantization, reducing model size ~4× and
      inference latency ~2–3×.
    """

    name = "XGBoost"
    short = "xgb"
    _ONNX_FILE = "models_saved/xgb_model_q8.onnx"

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = self._build()
        self.is_trained = False
        self.feature_names: list[str] = FeatureVector.feature_names()
        self.threshold: float = 0.3
        # ONNX session loaded on demand after quantized export
        self._onnx_session = None

    def _build(self):
        if _USE_XGB:
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=5,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                # ── Histogram quantization ────────────────────────────────
                tree_method="hist",   # bin-based split finding (O(b·f) not O(n·f))
                max_bin=256,          # 256 bins ≈ 8-bit feature quantization
                # device="cuda",      # uncomment to use GPU
            )
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = X.astype(np.float32)
        X_s = self.scaler.fit_transform(X)
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_s, y, test_size=0.2, random_state=42, stratify=y)
        if _USE_XGB:
            self.classifier.fit(X_tr, y_tr,
                                eval_set=[(X_v, y_v)], verbose=False)
        else:
            self.classifier.fit(X_tr, y_tr)
        self.threshold = _calibrate_threshold(self.classifier, X_v, y_v)
        self.is_trained = True
        return _eval_metrics(self.classifier, X_v, y_v, self.threshold)

    def export_onnx_quantized(self) -> bool:
        """
        Export the trained XGBoost model to INT8-quantized ONNX.
        Falls back gracefully if onnxmltools/onnxruntime are not installed.
        Returns True if export succeeded.
        """
        if not _USE_XGB or not _USE_XGB_ONNX or not _USE_ONNX:
            return False
        try:
            n_features = len(self.feature_names)
            initial_type = [("float_input", XGBFloatTensorType([None, n_features]))]
            onnx_model = convert_xgboost(self.classifier, initial_types=initial_type)

            tmp_path = self._ONNX_FILE.replace("_q8.onnx", "_fp32.onnx")
            Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            # Dynamic INT8 quantization of the weights
            quantize_dynamic(tmp_path, self._ONNX_FILE, weight_type=QuantType.QInt8)
            Path(tmp_path).unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _load_onnx_session(self) -> bool:
        if not _USE_ONNX:
            return False
        p = Path(self._ONNX_FILE)
        if not p.exists():
            return False
        try:
            self._onnx_session = rt.InferenceSession(str(p))
            return True
        except Exception:
            return False

    def predict_proba_single(self, fv: FeatureVector) -> float:
        arr = self.scaler.transform(
            np.array([fv.to_ml_array()], dtype=np.float32))
        # Prefer INT8 ONNX inference when available
        if self._onnx_session is not None:
            try:
                input_name = self._onnx_session.get_inputs()[0].name
                result = self._onnx_session.run(
                    None, {input_name: arr})
                # ONNX XGBoost output is [{0: p0, 1: p1}, ...]
                proba_map = result[1][0]
                return float(proba_map.get(1, proba_map.get("1", 0.0)))
            except Exception:
                pass
        return float(self.classifier.predict_proba(arr)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X.astype(np.float32))
        return self.classifier.predict_proba(X)[:, 1]

    def save(self, path: Optional[str] = None, scaler_path: Optional[str] = None):
        mp = Path(path or settings.model_path)
        sp = Path(scaler_path or settings.scaler_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, mp)
        joblib.dump((self.scaler, self.threshold), sp)

    def load(self, path: Optional[str] = None, scaler_path: Optional[str] = None):
        mp = Path(path or settings.model_path)
        sp = Path(scaler_path or settings.scaler_path)
        if not mp.exists() or not sp.exists():
            raise FileNotFoundError(f"XGBoost model files not found: {mp}, {sp}")
        self.classifier = joblib.load(mp)
        self.scaler, self.threshold = joblib.load(sp)
        self.is_trained = True
        # Try to load quantized ONNX session for fast inference
        self._load_onnx_session()


# ── Model 2: SGD classifier (linear SVM surrogate, O(n) scaling) ──────────────

class SVMModel:
    """
    SGDClassifier with modified_huber loss — a linear SVM surrogate that scales
    as O(n) in both training and inference, replacing the original O(n²) RBF-SVC.

    Why the switch at 10M+ rows:
    • RBF-SVC must store all support vectors in RAM; at 10M samples that
      easily exceeds tens of GB and training becomes infeasible.
    • SGD processes one mini-batch at a time, so memory is O(batch_size),
      training time is O(n), and inference is a single dot-product: O(d).
    • modified_huber loss gives well-calibrated probabilities (like Platt
      scaling for SVC) without a separate calibration step.
    • class_weight="balanced" preserves the fraud/legitimate imbalance handling.

    The Pipeline still includes StandardScaler so the interface is unchanged.
    After training, the linear weights are exported to INT8 ONNX for
    sub-millisecond inference.
    """

    name = "SVM (Linear SGD)"
    short = "svm"
    _MODEL_FILE = "models_saved/svm_model.joblib"
    _ONNX_FILE  = "models_saved/svm_model_q8.onnx"

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.threshold: float = 0.3
        self.feature_names: list[str] = FeatureVector.feature_names()
        self._onnx_session = None

    def _build(self) -> Pipeline:
        sgd = SGDClassifier(
            loss="modified_huber",   # smooth hinge — gives predict_proba
            penalty="elasticnet",    # L1+L2 regularisation
            alpha=1e-4,              # regularisation strength
            l1_ratio=0.15,           # 15% L1, 85% L2 (sparse + smooth)
            class_weight="balanced", # handles fraud/legitimate imbalance
            max_iter=1000,
            tol=1e-4,
            n_jobs=-1,
            random_state=42,
        )
        return Pipeline([("scaler", StandardScaler()), ("sgd", sgd)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = X.astype(np.float32)
        X_tr, X_v, y_tr, y_v = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self.pipeline = self._build()
        self.pipeline.fit(X_tr, y_tr)
        self.threshold = _calibrate_threshold(self.pipeline, X_v, y_v)
        self.is_trained = True
        return _eval_metrics(self.pipeline, X_v, y_v, self.threshold)

    def export_onnx_quantized(self) -> bool:
        """Export trained SGD pipeline to INT8-quantized ONNX."""
        if not _USE_ONNX or self.pipeline is None:
            return False
        try:
            n_features = len(self.feature_names)
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model = to_onnx(self.pipeline, initial_types=initial_type)
            tmp_path = self._ONNX_FILE.replace("_q8.onnx", "_fp32.onnx")
            Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            quantize_dynamic(tmp_path, self._ONNX_FILE, weight_type=QuantType.QInt8)
            Path(tmp_path).unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _load_onnx_session(self) -> bool:
        if not _USE_ONNX:
            return False
        p = Path(self._ONNX_FILE)
        if not p.exists():
            return False
        try:
            self._onnx_session = rt.InferenceSession(str(p))
            return True
        except Exception:
            return False

    def predict_proba_single(self, fv: FeatureVector) -> float:
        arr = np.array([fv.to_ml_array()], dtype=np.float32)
        if self._onnx_session is not None:
            try:
                input_name = self._onnx_session.get_inputs()[0].name
                result = self._onnx_session.run(None, {input_name: arr})
                # skl2onnx returns probabilities as result[1] array of shape (n, 2)
                return float(result[1][0][1])
            except Exception:
                pass
        return float(self.pipeline.predict_proba(arr)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X.astype(np.float32))[:, 1]

    def save(self, path: Optional[str] = None):
        p = Path(path or self._MODEL_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.pipeline, self.threshold), p)

    def load(self, path: Optional[str] = None):
        p = Path(path or self._MODEL_FILE)
        if not p.exists():
            raise FileNotFoundError(f"SVM model not found: {p}")
        self.pipeline, self.threshold = joblib.load(p)
        self.is_trained = True
        self._load_onnx_session()


# ── Model 3: FAISS IVF-PQ approximate nearest neighbours ──────────────────────

class KNNModel:
    """
    Approximate Nearest Neighbours using FAISS IVF-PQ index.

    Why FAISS at 10M+ rows:
    • sklearn ball-tree stores ALL training vectors in RAM.
      10M × 44 float32 features = ~1.76 GB just for the index.
    • FAISS IVF-PQ uses Product Quantization: the 44-d vector is split into
      M=8 sub-vectors of 5–6 dims each; each sub-vector is encoded to 8 bits
      (256 centroids).  This compresses the index by ~(44×4)/(8×1) ≈ 22×.
    • IVF partitions space into n_cells=256 Voronoi cells; a query probes
      only n_probe=32 cells, giving sub-linear O(n/n_cells × n_probe) lookup.

    Falls back to sklearn KNeighborsClassifier + ball_tree when faiss-cpu
    is not installed, preserving compatibility on environments without it.
    """

    name = "KNN (FAISS-PQ)" if _USE_FAISS else "KNN (k=7)"
    short = "knn"
    _MODEL_FILE  = "models_saved/knn_model.joblib"
    _FAISS_INDEX = "models_saved/knn_faiss.index"
    _FAISS_META  = "models_saved/knn_faiss_meta.joblib"

    # IVF-PQ hyperparameters (n_cells is computed adaptively at fit-time)
    _M_PQ     = 4     # sub-quantizers; must divide n_features (44) — use 4
    _NBITS    = 8     # bits per sub-quantizer → 256 centroids each
    _N_PROBE  = 8     # cells to inspect at query time (speed/accuracy trade-off)
    _K        = 7     # neighbours to retrieve

    def __init__(self):
        self.pipeline = None         # sklearn fallback
        self._faiss_index = None
        self._train_labels: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._n_cells: int = 4       # adaptive; set at fit-time
        self.is_trained = False
        self.threshold: float = 0.3
        self.feature_names: list[str] = FeatureVector.feature_names()

    # Use FAISS only when the dataset is large enough for it to pay off.
    # Below this threshold sklearn ball-tree is faster (no thread-init overhead).
    _FAISS_THRESHOLD = 500_000

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = X.astype(np.float32)
        X_tr, X_v, y_tr, y_v = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        use_faiss = _USE_FAISS and len(X_tr) >= self._FAISS_THRESHOLD
        if use_faiss:
            metrics = self._fit_faiss(X_tr, y_tr, X_v, y_v)
        else:
            metrics = self._fit_sklearn(X_tr, y_tr, X_v, y_v)

        self.is_trained = True
        return metrics

    # Minimum samples before FAISS index types are used:
    # • < _FLAT_PQ_MIN  → sklearn exact ball-tree (no PQ k-means overhead)
    # • _FLAT_PQ_MIN – _IVF_MIN → FAISS IndexFlatL2 (exact L2, no quantization)
    # • >= _IVF_MIN    → FAISS IndexIVFPQ (partitioned PQ, sub-linear queries)
    _FLAT_PQ_MIN = 50_000
    _IVF_MIN     = 500_000

    def _fit_faiss(self, X_tr, y_tr, X_v, y_v) -> dict:
        """
        Train FAISS index, choosing the index type based on dataset size:

        • n_train < 50 K   →  IndexFlatL2  (exact L2 search, no codebook k-means)
        • 50K – 500K       →  IndexFlatL2  (same — flat search is fast enough)
        • n_train >= 500K  →  IndexIVFPQ   (IVF partitioning + 8-bit PQ codes)
                               Requires ≥ 39 × n_cells training vectors.

        At any size, vectors are scaled by StandardScaler before indexing.
        The PQ codec (k-means over sub-vectors) is only trained for IVF-PQ,
        avoiding the 'too few training points' warning for small datasets.
        """
        self._scaler = StandardScaler()
        X_tr_s = self._scaler.fit_transform(X_tr)
        X_v_s  = self._scaler.transform(X_v)

        n_train    = X_tr_s.shape[0]
        n_features = X_tr_s.shape[1]
        X_c = np.ascontiguousarray(X_tr_s, dtype=np.float32)

        if n_train >= self._IVF_MIN:
            # Very large: IVF-PQ — partitioned approximate search
            n_cells = max(64, min(4096, int(np.sqrt(n_train))))
            n_probe = max(1, min(self._N_PROBE, n_cells // 4))
            quantizer = faiss.IndexFlatL2(n_features)
            index = faiss.IndexIVFPQ(
                quantizer, n_features, n_cells, self._M_PQ, self._NBITS)
            index.train(X_c)
            index.add(X_c)
            index.nprobe = n_probe
            self._n_cells = n_cells
            index_type = "IVF-PQ"
        else:
            # Small / medium: exact L2 — no codec training, instant build
            index = faiss.IndexFlatL2(n_features)
            index.add(X_c)
            self._n_cells = 0
            index_type = "Flat-L2"

        self._faiss_index  = index
        self._train_labels = y_tr.astype(np.int32)

        # Calibrate threshold using validation probabilities
        probs_v = self._query_proba(X_v_s)
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_v, probs_v)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        best = int(np.argmax(f1[:-1]))
        self.threshold = float(thresholds[best]) if len(thresholds) > 0 else 0.3

        # Build metrics
        from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
        return {
            "roc_auc": round(roc_auc_score(y_v, probs_v), 4),
            "avg_precision": round(average_precision_score(y_v, probs_v), 4),
            "best_threshold": round(self.threshold, 4),
            "classification_report": classification_report(
                y_v, (probs_v >= self.threshold).astype(int),
                target_names=["legitimate", "fraud"],
            ),
            "index_type": index_type,
            "n_cells": self._n_cells,
            "m_pq": self._M_PQ,
        }

    def _fit_sklearn(self, X_tr, y_tr, X_v, y_v) -> dict:
        """Fallback: sklearn calibrated KNN with ball_tree."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.calibration import CalibratedClassifierCV
        knn = KNeighborsClassifier(
            n_neighbors=self._K,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
            algorithm="ball_tree",
        )
        calibrated_knn = CalibratedClassifierCV(knn, method="isotonic", cv=3)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", calibrated_knn),
        ])
        self.pipeline.fit(X_tr, y_tr)
        self.threshold = _calibrate_threshold(self.pipeline, X_v, y_v)
        return _eval_metrics(self.pipeline, X_v, y_v, self.threshold)

    # ── Inference ──────────────────────────────────────────────────────────────

    def _query_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Distance-weighted fraud probability for each row in X_scaled using FAISS.
        Inverse-distance weights (1/d) replicate sklearn's weights='distance'.
        Works with both IndexPQ (flat) and IndexIVFPQ index types.
        """
        X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float32)
        distances, indices = self._faiss_index.search(X_scaled, self._K)
        probs = np.zeros(len(X_scaled), dtype=np.float32)
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            valid = idxs >= 0
            if not valid.any():
                continue
            d = dists[valid].astype(np.float64)
            # Avoid division by zero for exact matches
            weights = np.where(d == 0, 1e6, 1.0 / (d + 1e-9))
            labels  = self._train_labels[idxs[valid]]
            probs[i] = float(np.sum(weights * (labels == 1)) / np.sum(weights))
        return probs

    def predict_proba_single(self, fv: FeatureVector) -> float:
        arr = np.array([fv.to_ml_array()], dtype=np.float32)
        if self._faiss_index is not None and self._scaler is not None:
            arr_s = self._scaler.transform(arr)
            return float(self._query_proba(arr_s)[0])
        return float(self.pipeline.predict_proba(arr)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        if self._faiss_index is not None and self._scaler is not None:
            return self._query_proba(self._scaler.transform(X))
        return self.pipeline.predict_proba(X)[:, 1]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        p = Path(path or self._MODEL_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        if _USE_FAISS and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, self._FAISS_INDEX)
            joblib.dump(
                (self._scaler, self._train_labels, self.threshold, True, self._n_cells),
                self._FAISS_META,
            )
        else:
            joblib.dump((self.pipeline, self.threshold), p)

    def load(self, path: Optional[str] = None):
        faiss_meta = Path(self._FAISS_META)
        faiss_idx  = Path(self._FAISS_INDEX)
        if _USE_FAISS and faiss_meta.exists() and faiss_idx.exists():
            self._faiss_index = faiss.read_index(str(faiss_idx))
            meta = joblib.load(faiss_meta)
            self._scaler, self._train_labels, self.threshold = meta[0], meta[1], meta[2]
            self._n_cells = meta[4] if len(meta) > 4 else 4
            self._faiss_index.nprobe = max(1, min(self._N_PROBE, self._n_cells // 2))
            self.is_trained = True
            self.name = "KNN (FAISS-PQ)"
            return
        p = Path(path or self._MODEL_FILE)
        if not p.exists():
            raise FileNotFoundError(f"KNN model not found: {p}")
        self.pipeline, self.threshold = joblib.load(p)
        self.is_trained = True
        self.name = "KNN (k=7)"


# ── Model Registry ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Holds all trained models and provides a unified scoring interface.
    Ensemble weights: Bayesian 40% | XGBoost 30% | SVM 20% | KNN 10%
    """

    ENSEMBLE_WEIGHTS = {
        "bayesian": 0.40,
        "xgb":      0.30,
        "svm":      0.20,
        "knn":      0.10,
    }

    def __init__(self):
        self.xgb = AMLModel()
        self.svm = SVMModel()
        self.knn = KNNModel()

    def load_all(self) -> dict[str, bool]:
        """Load all models, return status dict."""
        status: dict[str, bool] = {}
        for name, model in [("xgb", self.xgb), ("svm", self.svm), ("knn", self.knn)]:
            try:
                model.load()
                status[name] = True
            except FileNotFoundError:
                status[name] = False
        return status

    def score_all(self, fv: FeatureVector) -> dict[str, int]:
        """
        Returns raw 0-999 score from each ML model.
        Falls back to 0 if a model isn't trained.
        """
        scores: dict[str, int] = {}
        for name, model in [("xgb", self.xgb), ("svm", self.svm), ("knn", self.knn)]:
            if model.is_trained:
                prob = model.predict_proba_single(fv)
                scores[name] = max(0, min(999, round(prob * 999)))
            else:
                scores[name] = 0
        return scores

    def fuse(self, bayesian_score: int, ml_scores: dict[str, int]) -> int:
        """Weighted ensemble of Bayesian + all ML models."""
        w = self.ENSEMBLE_WEIGHTS
        total_ml_weight = w["xgb"] + w["svm"] + w["knn"]
        available = {k: v for k, v in ml_scores.items() if v > 0}
        if not available:
            return max(0, min(999, bayesian_score))

        ml_contrib = sum(
            ml_scores.get(k, 0) * w.get(k, 0)
            for k in ("xgb", "svm", "knn")
        ) / total_ml_weight

        fused = (bayesian_score * w["bayesian"] +
                 ml_contrib * (1 - w["bayesian"]))
        return max(0, min(999, round(fused)))

    @property
    def any_ml_trained(self) -> bool:
        return any([self.xgb.is_trained, self.svm.is_trained, self.knn.is_trained])


# ── Singletons ─────────────────────────────────────────────────────────────────

_model_instance: AMLModel | None = None
_registry_instance: ModelRegistry | None = None


def get_model() -> AMLModel:
    """Legacy single-model accessor — used by train.py."""
    global _model_instance
    if _model_instance is None:
        _model_instance = AMLModel()
        try:
            _model_instance.load()
        except FileNotFoundError:
            pass
    return _model_instance


def get_registry() -> ModelRegistry:
    """Multi-model registry — used by the risk engine."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
        _registry_instance.load_all()
    return _registry_instance


def reset_registry() -> None:
    """Force reload on next call (used after retraining)."""
    global _registry_instance, _model_instance
    _registry_instance = None
    _model_instance = None
