"""Tests for pipeline.backends: rule baseline behavior + prompt versioning."""

import pytest

from pipeline.backends import (
    LABELS,
    REVIEW_LABELS,
    RuleBackend,
    get_backend,
    load_prompt,
)


def test_rule_backend_obvious_diabetes():
    backend = RuleBackend()
    (ann,) = backend.annotate(
        [
            (
                "c1",
                "Insulin resistance and glycemic control in type 2 diabetes. "
                "HbA1c was the primary outcome.",
            )
        ],
        LABELS,
        "v1",
    )
    assert ann.label == "diabetes"
    assert ann.parsed_ok is True
    assert 0.0 <= ann.confidence <= 1.0
    assert ann.backend == "rule"
    assert len(ann.prompt_sha256) == 64


def test_rule_backend_obvious_hypertension():
    backend = RuleBackend()
    (ann,) = backend.annotate(
        [
            (
                "c1",
                "Ambulatory blood pressure monitoring showed systolic pressure "
                "of 152 mmHg with a non-dipping pattern.",
            )
        ],
        LABELS,
        "v1",
    )
    assert ann.label == "hypertension"


def test_rule_backend_abstains_on_empty():
    backend = RuleBackend()
    (ann,) = backend.annotate([("c1", "   ")], LABELS, "v1")
    assert ann.label is None
    assert ann.parsed_ok is False


def test_rule_backend_abstains_without_keywords():
    backend = RuleBackend()
    (ann,) = backend.annotate(
        [("c1", "The weather was pleasant and the birds sang.")], LABELS, "v1"
    )
    assert ann.label is None
    assert ann.parsed_ok is False


def test_rule_backend_reviews():
    backend = RuleBackend()
    (ann,) = backend.annotate(
        [("c1", "I love this product, it is excellent and perfect.")],
        REVIEW_LABELS,
        "v1",
    )
    assert ann.label == "positive"


def test_rule_backend_deterministic():
    backend = RuleBackend()
    text = "Metformin and insulin are used in diabetes care daily."
    first = backend.annotate([("c1", text)], LABELS, "v1")[0]
    second = backend.annotate([("c1", text)], LABELS, "v1")[0]
    assert (first.label, first.confidence) == (second.label, second.confidence)


def test_load_prompt_versions_differ():
    t1, sha1 = load_prompt("v1")
    t2, sha2 = load_prompt("v2")
    assert "{labels}" in t1 and "{text}" in t1
    assert sha1 != sha2  # v2 adds the tie-break rule
    _, sha1_again = load_prompt("v1")
    assert sha1 == sha1_again  # stable hash


def test_load_prompt_unknown_version():
    with pytest.raises(FileNotFoundError, match="available"):
        load_prompt("v99")


def test_get_backend_unknown():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nope")


def test_get_backend_names():
    assert get_backend("rule").name == "rule"
    assert get_backend("gemini").name == "gemini"
