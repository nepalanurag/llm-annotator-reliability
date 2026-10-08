"""Experiment tracking: every annotation run is logged, reproducibly.

SQLite log (default pipeline/runs/experiments.db) with an explicit schema
version, plus JSONL export per run. Each entry records: backend name/version,
model id, prompt file + sha256, batch id, label set, full batch config, seed,
and every agreement metric with CIs. Offline-first by design: no server, no
credentials, works in CI.

Schema:
  runs(run_id, started_at, backend, backend_version, model_id, prompt_path,
       prompt_sha256, batch_id, n_items, labels_json, config_json, seed,
       schema_version)
  metrics(run_id, name, value, ci_lo, ci_hi)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import uuid

from pipeline.config import load_settings
from pipeline.logging import configure_logging, get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    backend TEXT NOT NULL,
    backend_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_path TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    n_items INTEGER NOT NULL,
    labels_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    seed INTEGER,
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    name TEXT NOT NULL,
    value REAL NOT NULL,
    ci_lo REAL,
    ci_hi REAL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open the experiment DB, creating schema on first use."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
        conn.commit()
    elif int(row[0]) != SCHEMA_VERSION:
        raise RuntimeError(
            f"experiment DB schema v{row[0]} != code v{SCHEMA_VERSION}; "
            "migrate the DB or point ANNOT_EXPERIMENT_DB at a fresh path"
        )
    return conn


def log_run(
    conn: sqlite3.Connection,
    *,
    backend: str,
    backend_version: str,
    model_id: str,
    prompt_path: str,
    prompt_sha256: str,
    batch_id: str,
    n_items: int,
    labels: list[str],
    config: dict,
    metrics: dict[str, tuple[float, float | None, float | None]],
    seed: int | None = None,
    run_id: str | None = None,
) -> str:
    """Log one annotation run. Returns the run_id."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if len(prompt_sha256) != 64:
        raise ValueError("prompt_sha256 must be a 64-char hex digest")
    run_id = run_id or uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            dt.datetime.now(dt.timezone.utc).isoformat(),
            backend,
            backend_version,
            model_id,
            prompt_path,
            prompt_sha256,
            batch_id,
            n_items,
            json.dumps(labels),
            json.dumps(config, sort_keys=True),
            seed,
            SCHEMA_VERSION,
        ),
    )
    for name, (value, lo, hi) in metrics.items():
        conn.execute(
            "INSERT INTO metrics VALUES (?,?,?,?,?)", (run_id, name, value, lo, hi)
        )
    conn.commit()
    log.info(
        "experiment_logged", run_id=run_id, batch_id=batch_id, n_metrics=len(metrics)
    )
    return run_id


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT run_id, started_at, backend, model_id, prompt_path, batch_id, n_items"
        " FROM runs ORDER BY started_at DESC"
    ).fetchall()
    return [
        dict(
            zip(
                [
                    "run_id",
                    "started_at",
                    "backend",
                    "model_id",
                    "prompt_path",
                    "batch_id",
                    "n_items",
                ],
                r,
            )
        )
        for r in rows
    ]


def show_run(conn: sqlite3.Connection, run_id: str) -> dict:
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"no run {run_id!r} in the experiment log")
    cols = [d[0] for d in conn.execute("SELECT * FROM runs LIMIT 0").description]
    run = dict(zip(cols, row))
    run["labels"] = json.loads(run.pop("labels_json"))
    run["config"] = json.loads(run.pop("config_json"))
    run["metrics"] = [
        {"name": n, "value": v, "ci_lo": lo, "ci_hi": hi}
        for n, v, lo, hi in conn.execute(
            "SELECT name, value, ci_lo, ci_hi FROM metrics WHERE run_id=?", (run_id,)
        ).fetchall()
    ]
    return run


def export_jsonl(conn: sqlite3.Connection, run_id: str, path: str) -> None:
    run = show_run(conn, run_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(run, indent=2) + "\n")
    log.info("experiment_exported", run_id=run_id, path=path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment log CLI.")
    parser.add_argument("--db", default=None, help="Experiment DB path")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List logged runs")
    show = sub.add_parser("show", help="Show one run in full")
    show.add_argument("run_id")
    exp = sub.add_parser("export", help="Export one run to JSONL")
    exp.add_argument("run_id")
    exp.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)
    conn = connect(args.db or settings.experiment_db)
    if args.cmd == "list":
        runs = list_runs(conn)
        print(json.dumps(runs, indent=2) if runs else "no runs logged yet")
    elif args.cmd == "show":
        try:
            print(json.dumps(show_run(conn, args.run_id), indent=2))
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.cmd == "export":
        try:
            export_jsonl(conn, args.run_id, args.out)
            print(f"exported {args.run_id} -> {args.out}")
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
