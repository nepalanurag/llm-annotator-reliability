"""Tests for pipeline.schemas: contract validation catches bad data."""

import pandas as pd
import pytest

from pipeline import CONTRACT_VERSION
from pipeline.schemas import (
    Annotation,
    Chunk,
    RawAbstract,
    validate_or_raise,
)
from pipeline.tests.fixtures import sample_abstracts


def test_contract_version_stamped():
    assert CONTRACT_VERSION == "1.0"


def test_raw_abstract_valid():
    df = validate_or_raise(RawAbstract, sample_abstracts())
    assert len(df) == 6


def test_raw_abstract_rejects_empty_text():
    df = sample_abstracts()
    df.loc[0, "text"] = None
    with pytest.raises(ValueError, match="Data contract violation"):
        validate_or_raise(RawAbstract, df)


def test_raw_abstract_rejects_duplicate_ids():
    df = sample_abstracts()
    df.loc[1, "doc_id"] = df.loc[0, "doc_id"]
    with pytest.raises(ValueError, match="Data contract violation"):
        validate_or_raise(RawAbstract, df)


def test_annotation_contract_rejects_bad_confidence():
    df = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "backend": "rule",
                "backend_version": "1.0",
                "model_id": "m",
                "prompt_version": "v1",
                "prompt_sha256": "a" * 64,
                "label": "diabetes",
                "confidence": 1.5,
                "parsed_ok": True,
                "latency_ms": 1.0,
                "input_tokens": None,
                "output_tokens": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="Data contract violation"):
        validate_or_raise(Annotation, df)


def test_chunk_contract_rejects_empty_chunk_id_set():
    df = pd.DataFrame(
        [
            {
                "doc_id": "d1",
                "chunk_id": "c1",
                "chunk_index": 0,
                "text": "x" * 50,
                "n_chars": 50,
                "boundary_ok": True,
                "ref_label": None,
            }
        ]
    )
    validate_or_raise(Chunk, df)  # valid
    df.loc[0, "n_chars"] = 0
    with pytest.raises(ValueError, match="Data contract violation"):
        validate_or_raise(Chunk, df)
