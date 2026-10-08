"""CLI wrapper: parquet of RawAbstract -> chunk JSONL.

Kept as a separate module so pipeline/chunking.py stays a pure library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from pipeline.chunking import Chunk, assert_boundaries_ok, chunk_text
from pipeline.config import load_settings
from pipeline.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk abstracts into JSONL.")
    parser.add_argument("input", help="Input parquet path (RawAbstract contract)")
    parser.add_argument("--out", required=True, help="Output chunk JSONL path")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)

    df = pd.read_parquet(args.input)
    chunks: list[Chunk] = []
    for _, row in df.iterrows():
        chunks.extend(
            chunk_text(
                str(row["doc_id"]),
                str(row["text"]),
                chunk_tokens=args.chunk_tokens or settings.chunk_tokens,
                overlap=(
                    args.overlap if args.overlap is not None else settings.chunk_overlap
                ),
                min_chunk_chars=settings.min_chunk_chars,
                ref_label=(
                    str(row["ref_label"]) if pd.notna(row.get("ref_label")) else None
                ),
            )
        )
    assert_boundaries_ok(chunks, min_chunk_chars=settings.min_chunk_chars)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "doc_id": c.doc_id,
                        "chunk_id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "ref_label": c.ref_label,
                    }
                )
                + "\n"
            )
    print(f"chunked {len(df)} abstracts -> {len(chunks)} chunks -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
