"""Tests for pipeline.chunking: sentence-aware splits + boundary checks."""

import pytest

from pipeline.chunking import (
    assert_boundaries_ok,
    check_boundaries,
    chunk_text,
    split_sentences,
    Chunk,
)


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []


def test_chunk_text_sentence_aligned():
    text = (
        "First sentence about diabetes and insulin resistance in adults. "
        "Second sentence about glucose control and monitoring at home. "
        "Third sentence about HbA1c outcomes after twelve long months."
    )
    chunks = chunk_text("d1", text, chunk_tokens=16, overlap=0.0)
    assert len(chunks) >= 2
    for ch in chunks:
        assert ch.text[-1] in ".!?"
    assert_boundaries_ok(chunks)  # must not raise


def test_chunk_text_overlap():
    text = " ".join(f"Sentence number {i} about diabetes care." for i in range(10))
    chunks = chunk_text("d1", text, chunk_tokens=40, overlap=0.5)
    assert len(chunks) >= 2
    # Overlap means some sentence appears in two consecutive chunks.
    frags = [[s for s in ch.text.split(". ") if s] for ch in chunks]
    assert any(set(frags[i]) & set(frags[i + 1]) for i in range(len(frags) - 1))


def test_chunk_text_rejects_empty():
    with pytest.raises(ValueError):
        chunk_text("d1", "   ")


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("d1", "Some text here.", overlap=0.95)


def test_boundary_check_flags_mid_sentence_start():
    bad = Chunk("d1", "c0", 0, "continuation without a capital start. Ends fine.")
    reports = check_boundaries([bad])
    assert not reports[0].ok
    assert any("mid-sentence" in r for r in reports[0].reasons)


def test_boundary_check_flags_missing_terminal():
    bad = Chunk("d1", "c0", 0, "Starts fine but never terminates properly")
    reports = check_boundaries([bad])
    assert not reports[0].ok


def test_assert_boundaries_ok_raises_loudly():
    bad = Chunk("d1", "c0", 0, "lowercase start and no terminal")
    with pytest.raises(ValueError, match="c0"):
        assert_boundaries_ok([bad])


def test_chunk_ids_unique_and_ordered():
    text = " ".join(f"Sentence {i} ends here." for i in range(8))
    chunks = chunk_text("doc9", text, chunk_tokens=12, overlap=0.0)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
