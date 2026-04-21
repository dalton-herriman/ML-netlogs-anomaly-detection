"""Application configuration loaded from environment / .env via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_path: str = Field(default="models/xgb_model.joblib")
    scaler_path: str = Field(default="models/scaler.joblib")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    mlflow_tracking_uri: str = Field(default="file:./mlruns")
    mlflow_experiment_name: str = Field(default="anomaly-detection")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
