"""Pipeline configuration via pydantic-settings.

Every knob is an environment variable with the ANNOT_ prefix, so the same
code runs locally, in Docker, and in CI without edits.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnnotSettings(BaseSettings):
    """All pipeline settings. Override with ANNOT_<NAME> env vars."""

    model_config = SettingsConfigDict(env_prefix="ANNOT_", extra="ignore")

    # Extraction
    arxiv_query: str = Field(
        default="ti:diabetes OR ti:hypertension OR ti:asthma OR ti:migraine",
        description="arXiv API search query for the extract stage.",
    )
    max_results: int = Field(default=50, ge=1, le=2000)
    request_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=4, ge=0, le=10)
    polite_delay_s: float = Field(default=3.0, ge=0)

    # Chunking / dedup
    chunk_tokens: int = Field(default=256, ge=16)
    chunk_overlap: float = Field(default=0.15, ge=0.0, lt=0.9)
    min_chunk_chars: int = Field(default=40, ge=1)
    dedup_threshold: float = Field(default=0.8, gt=0.0, lt=1.0)
    minhash_permutations: int = Field(default=128, ge=32)

    # Annotation
    backend: str = Field(default="rule", description="'rule' or 'gemini'")
    prompt_version: str = Field(default="v1", description="Prompt file version.")
    batch_size: int = Field(default=32, ge=1)

    # Drift monitoring
    psi_alert_threshold: float = Field(default=0.25, gt=0)
    psi_warning_threshold: float = Field(default=0.10, gt=0)
    chi2_alpha: float = Field(default=0.05, gt=0, lt=1)

    # Paths (relative to repo root unless absolute)
    batch_dir: str = Field(default="pipeline/batches")
    experiment_db: str = Field(default="pipeline/runs/experiments.db")
    baseline_path: str = Field(default="pipeline/baselines/label_baseline.json")

    # Logging
    log_format: str = Field(default="json")
    log_level: str = Field(default="INFO")

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        if v not in ("rule", "gemini"):
            raise ValueError("backend must be 'rule' or 'gemini'")
        return v

    @field_validator("log_format")
    @classmethod
    def _check_log_format(cls, v: str) -> str:
        if v not in ("json", "console"):
            raise ValueError("log_format must be 'json' or 'console'")
        return v


def load_settings() -> AnnotSettings:
    """Load settings, failing loudly on invalid values."""
    try:
        return AnnotSettings()
    except Exception as exc:  # pydantic ValidationError
        raise SystemExit(f"Invalid pipeline configuration: {exc}") from exc
