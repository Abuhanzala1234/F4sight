"""API settings. Secrets come from the environment, never from config/ (§11)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database ---
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "drishti"
    db_password: str = "drishti_dev"
    db_name: str = "drishti"

    # --- object storage ---
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "drishti"
    minio_secret_key: str = "drishti_dev_secret"
    minio_secure: bool = False
    minio_bucket_evidence: str = "drishti-evidence"
    minio_bucket_clips: str = "drishti-clips"

    # --- cache / bus ---
    redis_url: str = "redis://localhost:6379/0"
    alert_stream: str = "drishti:alerts"
    live_track_stream: str = "drishti:live"

    # --- media ---
    mediamtx_host: str = "localhost"
    hls_port: int = 8888
    webrtc_port: int = 8889
    rtsp_port: int = 8554
    mediamtx_api_port: int = 9997

    # --- crypto ---
    jwt_secret: str = Field(default="dev_only_change_me")
    jwt_algorithm: str = "HS256"
    access_token_ttl_s: int = 3600
    refresh_token_ttl_s: int = 604800
    plate_hmac_key: str = Field(default="dev_only_change_me")

    # --- behaviour ---
    cors_origins: list[str] = ["http://localhost:5173"]
    presigned_url_ttl_s: int = 300
    login_rate_limit_per_min: int = 5
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def hls_base(self) -> str:
        return f"http://{self.mediamtx_host}:{self.hls_port}"

    def warn_on_dev_secrets(self) -> list[str]:
        """Loud about insecure defaults. A demo running on 'dev_only_change_me'
        is fine; a BOP running on it is not, and nobody should have to read the
        source to find out which one they have."""
        problems = []
        if self.jwt_secret.startswith("dev_only"):
            problems.append("JWT_SECRET is the development default")
        if self.plate_hmac_key.startswith("dev_only"):
            problems.append("PLATE_HMAC_KEY is the development default")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
