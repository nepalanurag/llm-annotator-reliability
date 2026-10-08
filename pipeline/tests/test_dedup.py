"""Tests for pipeline.deduplication: exact + MinHash near-dup detection."""

import pytest

from pipeline.chunking import Chunk
from pipeline.deduplication import deduplicate, minhash_of, normalize, shingles
from pipeline.tests.fixtures import near_dup_pair


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(doc_id="d1", chunk_id=cid, chunk_index=0, text=text)


def test_exact_duplicate_dropped():
    text = "Insulin resistance and glycemic control in type 2 diabetes. HbA1c fell."
    result = deduplicate([_chunk("c1", text), _chunk("c2", text)])
    assert [c.chunk_id for c in result.kept] == ["c1"]
    assert result.exact_duplicates == ["c2"]
    assert result.stats()["n_exact_duplicates"] == 1


def test_exact_duplicate_ignores_whitespace():
    a = "Blood pressure control in hypertension.  Systolic fell below target."
    b = "Blood pressure control in hypertension. Systolic fell below target."
    result = deduplicate([_chunk("c1", a), _chunk("c2", b)])
    assert len(result.kept) == 1
    assert result.exact_duplicates == ["c2"]


def test_near_duplicate_caught():
    a, b = near_dup_pair()
    other = (
        "Migraine aura and photophobia are common. Triptans relieve attacks quickly."
    )
    result = deduplicate([_chunk("c1", a), _chunk("c2", b), _chunk("c3", other)])
    kept_ids = {c.chunk_id for c in result.kept}
    assert kept_ids == {"c1", "c3"}
    assert len(result.near_duplicates) == 1
    first, second, est = result.near_duplicates[0]
    assert (first, second) == ("c1", "c2")
    assert est > 0.8


def test_distinct_chunks_all_kept():
    texts = [
        "Insulin resistance and glycemic control in type 2 diabetes patients here.",
        "Ambulatory blood pressure monitoring shows a non-dipping pattern nightly.",
        "Inhaled corticosteroids improve asthma control questionnaire scores well.",
        "Triptans relieve migraine attacks within two hours for most patients.",
    ]
    result = deduplicate([_chunk(f"c{i}", t) for i, t in enumerate(texts)])
    assert len(result.kept) == 4
    assert result.stats()["n_near_duplicates"] == 0


def test_dedup_rejects_bad_threshold():
    with pytest.raises(ValueError):
        deduplicate(
            [_chunk("c1", "Some valid chunk text for testing purposes.")], threshold=1.5
        )


def test_normalize_and_shingles():
    assert normalize("  Hello   WORLD ") == "hello world"
    assert len(shingles("a b c d e f", k=5)) == 2
    assert shingles("", k=5) == set()


def test_minhash_deterministic():
    a, b = near_dup_pair()
    assert minhash_of(a).jaccard(minhash_of(a)) == 1.0
    assert minhash_of(a).jaccard(minhash_of(b)) == minhash_of(a).jaccard(minhash_of(b))
