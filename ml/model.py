"""
AML ML Models
=============
Three independent classifiers for risk scoring:

  1. XGBoost (AMLModel)  — gradient-boosted trees, best overall AUC
  2. SVM     (SVMModel)  — RBF kernel SVM, strong class-boundary detection
  3. KNN     (KNNModel)  — distance-weighted k-NN, similarity-based anomaly detection

Each model exposes a common interface:
  .fit(X, y)             → metrics dict
  .predict_proba_single(fv: FeatureVector) → float  (0.0 – 1.0)
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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _USE_XGB = False

from config.settings import settings
from risk.features import FeatureVector


# ── Shared helpers ────────────────────────────────────────────────────────────

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


# ── Model 1: XGBoost / Gradient Boosting ─────────────────────────────────────

class AMLModel:
    """Gradient-boosted trees (XGBoost with sklearn fallback)."""

    name = "XGBoost"
    short = "xgb"

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = self._build()
        self.is_trained = False
        self.feature_names: list[str] = FeatureVector.feature_names()
        self.threshold: float = 0.3

    def _build(self):
        if _USE_XGB:
            return XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=5, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
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

    def predict_proba_single(self, fv: FeatureVector) -> float:
        arr = self.scaler.transform(np.array([fv.to_ml_array()]))
        return float(self.classifier.predict_proba(arr)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(X))[:, 1]

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


# ── Model 2: Support Vector Machine ──────────────────────────────────────────

class SVMModel:
    """
    RBF-kernel SVM with Platt scaling (probability=True).
    Strong at finding non-linear decision boundaries in high-dimensional
    feature spaces — excellent at separating fraud clusters from normal behaviour.
    """

    name = "SVM"
    short = "svm"
    _MODEL_FILE = "models_saved/svm_model.joblib"

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.threshold: float = 0.3
        self.feature_names: list[str] = FeatureVector.feature_names()

    def _build(self) -> Pipeline:
        # StandardScaler is critical for SVM — features must be on the same scale
        svc = SVC(
            kernel="rbf",
            C=10.0,            # Regularisation — higher C fits training data tighter
            gamma="scale",     # Kernel bandwidth = 1 / (n_features * X.var())
            class_weight="balanced",   # Handles fraud:legitimate imbalance
            probability=True,          # Platt scaling for calibrated probabilities
            random_state=42,
            cache_size=500,
        )
        return Pipeline([("scaler", StandardScaler()), ("svm", svc)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        X_tr, X_v, y_tr, y_v = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self.pipeline = self._build()
        self.pipeline.fit(X_tr, y_tr)
        self.threshold = _calibrate_threshold(self.pipeline, X_v, y_v)
        self.is_trained = True
        return _eval_metrics(self.pipeline, X_v, y_v, self.threshold)

    def predict_proba_single(self, fv: FeatureVector) -> float:
        arr = np.array([fv.to_ml_array()])
        return float(self.pipeline.predict_proba(arr)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

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


# ── Model 3: K-Nearest Neighbours ────────────────────────────────────────────

class KNNModel:
    """
    Distance-weighted k-NN with isotonic calibration.
    Detects anomalies by proximity to known fraud cases in feature space.
    Captures localised fraud clusters that rule-based systems may miss.
    """

    name = "KNN"
    short = "knn"
    _MODEL_FILE = "models_saved/knn_model.joblib"

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.threshold: float = 0.3
        self.feature_names: list[str] = FeatureVector.feature_names()

    def _build(self) -> Pipeline:
        # Raw KNN probabilities need calibration — use isotonic regression
        knn = KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",   # Closer neighbours get higher weight
            metric="euclidean",
            n_jobs=-1,
            algorithm="ball_tree",
        )
        calibrated_knn = CalibratedClassifierCV(knn, method="isotonic", cv=3)
        return Pipeline([("scaler", StandardScaler()), ("knn", calibrated_knn)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        X_tr, X_v, y_tr, y_v = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self.pipeline = self._build()
        self.pipeline.fit(X_tr, y_tr)
        self.threshold = _calibrate_threshold(self.pipeline, X_v, y_v)
        self.is_trained = True
        return _eval_metrics(self.pipeline, X_v, y_v, self.threshold)

    def predict_proba_single(self, fv: FeatureVector) -> float:
        arr = np.array([fv.to_ml_array()])
        return float(self.pipeline.predict_proba(arr)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: Optional[str] = None):
        p = Path(path or self._MODEL_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.pipeline, self.threshold), p)

    def load(self, path: Optional[str] = None):
        p = Path(path or self._MODEL_FILE)
        if not p.exists():
            raise FileNotFoundError(f"KNN model not found: {p}")
        self.pipeline, self.threshold = joblib.load(p)
        self.is_trained = True


# ── Model Registry ────────────────────────────────────────────────────────────

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
        # Normalise ML weights if some models aren't available
        available = {k: v for k, v in ml_scores.items() if v > 0}
        if not available:
            # Fallback: pure Bayesian
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


# ── Singletons ────────────────────────────────────────────────────────────────

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
