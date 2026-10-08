"""Sentence-aware chunking with boundary sanity checks.

Abstracts are short, so the chunker is deliberately simple: split on sentence
terminals, greedily pack sentences up to a character budget derived from
chunk_tokens, and carry an overlap of trailing sentences into the next chunk.
Every chunk then passes boundary sanity checks; violations fail loudly rather
than silently producing bad training text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.logging import get_logger

log = get_logger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TERMINAL = (".", "!", "?", '"', "'", ")", "]")

# Rough chars-per-token for biomedical English; used only to size chunks.
_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    chunk_index: int
    text: str
    ref_label: str | None = None

    @property
    def n_chars(self) -> int:
        return len(self.text)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation."""
    parts = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    return parts


def chunk_text(
    doc_id: str,
    text: str,
    chunk_tokens: int = 256,
    overlap: float = 0.15,
    min_chunk_chars: int = 40,
    ref_label: str | None = None,
) -> list[Chunk]:
    """Split one document into overlapping, sentence-aligned chunks."""
    if not text or not text.strip():
        raise ValueError(f"empty text for doc {doc_id}")
    if not 0.0 <= overlap < 0.9:
        raise ValueError(f"overlap must be in [0, 0.9), got {overlap}")
    budget = int(chunk_tokens * _CHARS_PER_TOKEN)
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError(f"no sentences found for doc {doc_id}")

    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(sentences):
        # Greedily pack sentences up to the budget.
        j = i
        size = 0
        while j < len(sentences) and size + len(sentences[j]) + 1 <= budget:
            size += len(sentences[j]) + 1
            j += 1
        if j == i:  # one very long sentence: keep it whole, never split mid-sentence
            j = i + 1
        piece = " ".join(sentences[i:j]).strip()
        if len(piece) >= min_chunk_chars:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#c{idx}",
                    chunk_index=idx,
                    text=piece,
                    ref_label=ref_label,
                )
            )
            idx += 1
        # Overlap: step back by `overlap` fraction of the consumed sentences.
        consumed = j - i
        step = max(1, int(round(consumed * (1.0 - overlap))))
        i += step

    if not chunks:
        raise ValueError(f"chunking produced no chunks for doc {doc_id}")
    return chunks


@dataclass(frozen=True)
class BoundaryReport:
    chunk_id: str
    ok: bool
    reasons: tuple[str, ...]


def check_boundaries(
    chunks: list[Chunk],
    min_chunk_chars: int = 40,
    max_chunk_chars: int = 4000,
) -> list[BoundaryReport]:
    """Sanity-check every chunk boundary. Returns one report per chunk.

    Checks: (1) chunk starts at a sentence boundary (capital letter or digit,
    not a lowercase continuation); (2) chunk ends at a terminal punctuation
    mark; (3) length within [min, max]; (4) no empty chunks.
    """
    reports: list[BoundaryReport] = []
    for ch in chunks:
        reasons: list[str] = []
        t = ch.text.strip()
        if not t:
            reasons.append("empty chunk text")
        else:
            if len(t) < min_chunk_chars:
                reasons.append(f"too short ({len(t)} < {min_chunk_chars} chars)")
            if len(t) > max_chunk_chars:
                reasons.append(f"too long ({len(t)} > {max_chunk_chars} chars)")
            if t[-1] not in _TERMINAL:
                reasons.append(f"does not end at sentence terminal: ...{t[-30:]!r}")
            first = t[0]
            if first.islower():
                reasons.append(f"starts mid-sentence (lowercase {first!r})")
        reports.append(BoundaryReport(ch.chunk_id, not reasons, tuple(reasons)))
    return reports


def assert_boundaries_ok(
    chunks: list[Chunk],
    min_chunk_chars: int = 40,
    max_chunk_chars: int = 4000,
) -> None:
    """Raise a loud error listing every chunk that fails boundary checks."""
    reports = check_boundaries(chunks, min_chunk_chars, max_chunk_chars)
    bad = [r for r in reports if not r.ok]
    if bad:
        detail = "\n".join(f"  {r.chunk_id}: {'; '.join(r.reasons)}" for r in bad)
        raise ValueError(
            f"{len(bad)}/{len(chunks)} chunks failed boundary checks:\n{detail}"
        )
    log.info("chunk_boundary_check_passed", n_chunks=len(chunks))
