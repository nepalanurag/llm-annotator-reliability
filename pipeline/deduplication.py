"""Dedup + chunking quality gate before annotation.

Two layers:
1. Exact duplicates via sha256 of normalized text (free, catches re-exports).
2. Near-duplicates via MinHash LSH over word 5-grams (datasketch), Jaccard
   threshold configurable (default 0.8). Near-dup pairs are logged, not
   silently dropped: the first occurrence is kept, later ones quarantined.

Also runs the chunk-boundary sanity checks from pipeline.chunking and folds
everything into one dedup stats dict per batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from pipeline import CONTRACT_VERSION
from pipeline.chunking import Chunk, assert_boundaries_ok
from pipeline.config import load_settings
from pipeline.logging import configure_logging, get_logger

log = get_logger(__name__)

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace for comparison."""
    return _WS.sub(" ", text.strip().lower())


def shingles(text: str, k: int = 5) -> set[str]:
    """Word k-gram shingles of normalized text."""
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def minhash_of(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm, seed=42)
    for s in shingles(text):
        mh.update(s.encode("utf-8"))
    return mh


@dataclass
class DedupResult:
    kept: list[Chunk] = field(default_factory=list)
    exact_duplicates: list[str] = field(default_factory=list)  # chunk_ids dropped
    near_duplicates: list[tuple[str, str, float]] = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "n_input": len(self.kept)
            + len(self.exact_duplicates)
            + len({b for _, b, _ in self.near_duplicates}),
            "n_kept": len(self.kept),
            "n_exact_duplicates": len(self.exact_duplicates),
            "n_near_duplicates": len(self.near_duplicates),
        }


def deduplicate(
    chunks: list[Chunk],
    threshold: float = 0.8,
    num_perm: int = 128,
    min_chunk_chars: int = 40,
) -> DedupResult:
    """Run boundary checks, then exact + MinHash dedup. Returns kept chunks."""
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    assert_boundaries_ok(chunks, min_chunk_chars=min_chunk_chars)

    result = DedupResult()
    seen_hashes: dict[str, str] = {}
    minhashes: dict[str, MinHash] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for ch in chunks:
        digest = hashlib.sha256(normalize(ch.text).encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            result.exact_duplicates.append(ch.chunk_id)
            log.info(
                "exact_duplicate",
                chunk_id=ch.chunk_id,
                first_seen=seen_hashes[digest],
            )
            continue
        mh = minhash_of(ch.text, num_perm=num_perm)
        dupes = lsh.query(mh)
        if dupes:
            est = round(float(mh.jaccard(minhashes[dupes[0]])), 3)
            result.near_duplicates.append((dupes[0], ch.chunk_id, est))
            log.info(
                "near_duplicate",
                chunk_id=ch.chunk_id,
                matches=dupes,
                jaccard_est=est,
                threshold=threshold,
            )
            continue
        seen_hashes[digest] = ch.chunk_id
        minhashes[ch.chunk_id] = mh
        lsh.insert(ch.chunk_id, mh)
        result.kept.append(ch)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deduplicate chunk JSONL (fields: doc_id, chunk_id, chunk_index, text)."
    )
    parser.add_argument("input", help="Input JSONL path with chunk records")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out", default=None, help="Write kept chunks JSONL here")
    parser.add_argument("--stats", default=None, help="Write dedup stats JSON here")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)
    threshold = (
        args.threshold if args.threshold is not None else settings.dedup_threshold
    )

    chunks: list[Chunk] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                chunks.append(
                    Chunk(r["doc_id"], r["chunk_id"], int(r["chunk_index"]), r["text"])
                )
    if not chunks:
        print("error: no chunks in input", file=sys.stderr)
        return 2
    result = deduplicate(
        chunks,
        threshold=threshold,
        num_perm=settings.minhash_permutations,
        min_chunk_chars=settings.min_chunk_chars,
    )
    stats = result.stats()
    log.info("dedup_complete", **stats)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for ch in result.kept:
                f.write(
                    json.dumps(
                        {
                            "doc_id": ch.doc_id,
                            "chunk_id": ch.chunk_id,
                            "chunk_index": ch.chunk_index,
                            "text": ch.text,
                        }
                    )
                    + "\n"
                )
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
