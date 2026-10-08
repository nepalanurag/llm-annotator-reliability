"""Annotate stage CLI: chunked texts -> backend -> versioned batch.

Reads chunk JSONL (from the chunk/dedup stages), runs the chosen backend,
validates against the Annotation contract, and writes a versioned batch dir:

    pipeline/batches/<batch_id>/
        manifest.json        # batch_id, backend, prompt sha, config, counts
        annotations.parquet  # one row per chunk, Annotation contract

Batches are the DVC-tracked unit: `dvc add pipeline/batches/<batch_id>`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import uuid

import pandas as pd

from pipeline import CONTRACT_VERSION
from pipeline.backends import LABELS, get_backend, load_prompt
from pipeline.chunking import Chunk
from pipeline.config import load_settings
from pipeline.logging import configure_logging, get_logger
from pipeline.schemas import validate_or_raise, Annotation as AnnotationSchema

log = get_logger(__name__)


def run_batch(
    chunks: list[Chunk],
    labels: list[str],
    backend_name: str,
    prompt_version: str,
    batch_id: str,
    batch_dir: str,
) -> dict:
    """Annotate chunks and write the versioned batch. Returns the manifest."""
    if not chunks:
        raise ValueError("no chunks to annotate")
    if not labels:
        raise ValueError("label list must not be empty")
    backend = get_backend(backend_name)
    _, prompt_sha = load_prompt(prompt_version)

    log.info(
        "annotation_start",
        batch_id=batch_id,
        backend=backend_name,
        prompt_version=prompt_version,
        n_chunks=len(chunks),
    )
    records = backend.annotate(
        [(c.chunk_id, c.text) for c in chunks], labels, prompt_version
    )
    rows = [
        {
            "chunk_id": r.chunk_id,
            "backend": r.backend,
            "backend_version": r.backend_version,
            "model_id": r.model_id,
            "prompt_version": r.prompt_version,
            "prompt_sha256": r.prompt_sha256,
            "label": r.label,
            "confidence": r.confidence,
            "parsed_ok": r.parsed_ok,
            "latency_ms": r.latency_ms,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    # Token counts are absent for non-API backends: use nullable Int64 so the
    # contract sees integers-or-null, not an object column.
    for col in ("input_tokens", "output_tokens"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = validate_or_raise(AnnotationSchema, df)

    out_dir = os.path.join(batch_dir, batch_id)
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "annotations.parquet"), index=False)
    manifest = {
        "batch_id": batch_id,
        "contract_version": CONTRACT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": backend_name,
        "backend_version": backend.version,
        "model_id": backend.model_id,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha,
        "labels": labels,
        "n_chunks": len(chunks),
        "n_annotated": int(df["parsed_ok"].sum()),
        "n_abstentions": int((~df["parsed_ok"]).sum()),
        "mean_latency_ms": round(float(df["latency_ms"].mean()), 2),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info(
        "annotation_complete",
        **{k: v for k, v in manifest.items() if k != "prompt_sha256"},
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-annotate chunks with a pluggable backend."
    )
    parser.add_argument("chunks", help="Input chunk JSONL path")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["rule", "gemini"],
        help="Annotation backend (default: config)",
    )
    parser.add_argument(
        "--labels", default=",".join(LABELS), help="Comma-separated label set"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        dest="prompt_version",
        help="Prompt version, e.g. v1 (default: config)",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Batch id (default: UTC timestamp + short uuid)",
    )
    parser.add_argument("--batch-dir", default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)

    backend_name = args.backend or settings.backend
    prompt_version = args.prompt_version or settings.prompt_version
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        print("error: --labels must list at least one label", file=sys.stderr)
        return 2
    batch_id = args.batch_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    batch_dir = args.batch_dir or settings.batch_dir

    chunks: list[Chunk] = []
    with open(args.chunks, encoding="utf-8") as f:
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
    # Fail fast on unknown prompt versions before any annotation work.
    load_prompt(prompt_version)
    manifest = run_batch(
        chunks, labels, backend_name, prompt_version, batch_id, batch_dir
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
