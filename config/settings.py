from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="amlpassword123", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    risk_allow_max: int = Field(default=399, alias="RISK_ALLOW_MAX")
    risk_challenge_max: int = Field(default=699, alias="RISK_CHALLENGE_MAX")

    base_fraud_rate: float = Field(default=0.02, alias="BASE_FRAUD_RATE")

    model_path: str = Field(default="models_saved/aml_model.joblib", alias="MODEL_PATH")
    scaler_path: str = Field(default="models_saved/aml_scaler.joblib", alias="SCALER_PATH")

    # ── Feature flags ──────────────────────────────────────────────────────────
    # Enable KNN (FAISS) anomaly detector training and inference.
    # Disable on memory-constrained machines or when GraphSAGE is preferred.
    enable_knn_anomaly: bool = Field(default=True, alias="ENABLE_KNN_ANOMALY")

    # Enable GraphSAGE mule-account detection.
    enable_graphsage: bool = Field(default=True, alias="ENABLE_GRAPHSAGE")

    model_config = {"env_file": ".env", "populate_by_name": True}

    @property
    def outcome_thresholds(self) -> dict:
        return {
            "ALLOW": (0, self.risk_allow_max),
            "CHALLENGE": (self.risk_allow_max + 1, self.risk_challenge_max),
            "DECLINE": (self.risk_challenge_max + 1, 999),
        }


settings = Settings()
