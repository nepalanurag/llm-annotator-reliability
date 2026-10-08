"""Explicit, versioned data contracts for every pipeline stage.

Each stage validates its output against the schema below before passing it
downstream. CONTRACT_VERSION (pipeline/__init__.py) is stamped into batch
manifests; bump it there on any schema change.
"""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series

LABELS = ["diabetes", "hypertension", "asthma", "migraine"]


class RawAbstract(pa.DataFrameModel):
    """Output of the extract stage: one row per source abstract."""

    doc_id: Series[str] = pa.Field(nullable=False, unique=True)
    title: Series[str] = pa.Field(nullable=True)
    text: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(nullable=False)  # "arxiv" | "csv"
    retrieved_at: Series[str] = pa.Field(nullable=False)  # ISO-8601 UTC
    ref_label: Series[str] = pa.Field(
        nullable=True
    )  # distant reference label, if known

    class Config:
        strict = True


class Chunk(pa.DataFrameModel):
    """Output of the chunk stage: sentence-aware chunks with overlap."""

    doc_id: Series[str] = pa.Field(nullable=False)
    chunk_id: Series[str] = pa.Field(nullable=False, unique=True)
    chunk_index: Series[int] = pa.Field(ge=0)
    text: Series[str] = pa.Field(nullable=False)
    n_chars: Series[int] = pa.Field(gt=0)
    boundary_ok: Series[bool] = pa.Field(nullable=False)
    ref_label: Series[str] = pa.Field(nullable=True)  # carried from RawAbstract

    class Config:
        strict = True


class Annotation(pa.DataFrameModel):
    """Output of the annotate stage: one row per (chunk, backend) call."""

    chunk_id: Series[str] = pa.Field(nullable=False)
    backend: Series[str] = pa.Field(isin=["rule", "gemini"])
    backend_version: Series[str] = pa.Field(nullable=False)
    model_id: Series[str] = pa.Field(nullable=False)
    prompt_version: Series[str] = pa.Field(nullable=False)
    prompt_sha256: Series[str] = pa.Field(nullable=False, str_length=64)
    label: Series[str] = pa.Field(nullable=True)  # None = abstention
    confidence: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=True)
    parsed_ok: Series[bool] = pa.Field(nullable=False)
    latency_ms: Series[float] = pa.Field(ge=0.0)
    input_tokens: Series[int] = pa.Field(ge=0, nullable=True)
    output_tokens: Series[int] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = True


class AgreementMetrics(pa.DataFrameModel):
    """One row per metric computed by the agreement stage."""

    batch_id: Series[str] = pa.Field(nullable=False)
    metric: Series[str] = pa.Field(nullable=False)
    value: Series[float] = pa.Field(nullable=False)
    ci_lo: Series[float] = pa.Field(nullable=True)
    ci_hi: Series[float] = pa.Field(nullable=True)
    n: Series[int] = pa.Field(gt=0)

    class Config:
        strict = True


def validate_or_raise(model: type[pa.DataFrameModel], df):  # type: ignore[valid-type]
    """Validate df against a contract, raising a loud error on violation."""
    try:
        return model.validate(df)
    except pa.errors.SchemaError as exc:
        raise ValueError(f"Data contract violation ({model.__name__}): {exc}") from exc
