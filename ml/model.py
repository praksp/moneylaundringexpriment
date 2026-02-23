"""
AML ML Model
=============
Gradient-Boosted Trees (XGBoost with sklearn fallback) trained on
graph-derived features + Bayesian scores from the seeded transaction dataset.

The ML model's output (fraud probability 0–1) is fused with the Bayesian
score in the risk engine using a weighted ensemble.
"""
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

try:
    from xgboost import XGBClassifier
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _USE_XGB = False

from config.settings import settings
from risk.features import FeatureVector


class AMLModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = self._build_classifier()
        self.is_trained = False
        self.feature_names: list[str] = FeatureVector.feature_names()
        self.threshold: float = 0.3  # Classification threshold tuned during training

    def _build_classifier(self):
        if _USE_XGB:
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=5,     # Handle class imbalance (1:5 ratio fraud:legit)
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the model. Returns evaluation metrics on a hold-out set."""
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        if _USE_XGB:
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.classifier.fit(X_train, y_train)

        self.is_trained = True

        # Calibrate threshold on validation set
        probs = self.classifier.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        # Find threshold that maximises F1
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = int(np.argmax(f1_scores[:-1]))
        self.threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.3

        metrics = {
            "roc_auc": round(roc_auc_score(y_val, probs), 4),
            "avg_precision": round(average_precision_score(y_val, probs), 4),
            "best_threshold": round(self.threshold, 4),
            "classification_report": classification_report(
                y_val, (probs >= self.threshold).astype(int),
                target_names=["legitimate", "fraud"],
            ),
        }
        return metrics

    def predict_proba_single(self, fv: FeatureVector) -> float:
        """Return fraud probability for a single FeatureVector (0.0–1.0)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Run training first.")
        arr = np.array([fv.to_ml_array()])
        arr_scaled = self.scaler.transform(arr)
        return float(self.classifier.predict_proba(arr_scaled)[0, 1])

    def predict_proba_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return array of fraud probabilities for a batch."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        scaled = self.scaler.transform(feature_matrix)
        return self.classifier.predict_proba(scaled)[:, 1]

    def save(self, model_path: str | None = None, scaler_path: str | None = None) -> None:
        mp = Path(model_path or settings.model_path)
        sp = Path(scaler_path or settings.scaler_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, mp)
        joblib.dump((self.scaler, self.threshold), sp)

    def load(self, model_path: str | None = None, scaler_path: str | None = None) -> None:
        mp = Path(model_path or settings.model_path)
        sp = Path(scaler_path or settings.scaler_path)
        if not mp.exists() or not sp.exists():
            raise FileNotFoundError(
                f"Model files not found at {mp} / {sp}. "
                "Run `python scripts/setup.py` first."
            )
        self.classifier = joblib.load(mp)
        self.scaler, self.threshold = joblib.load(sp)
        self.is_trained = True


# Singleton instance loaded once
_model_instance: AMLModel | None = None


def get_model() -> AMLModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = AMLModel()
        try:
            _model_instance.load()
        except FileNotFoundError:
            pass  # Model will be trained / loaded later
    return _model_instance
