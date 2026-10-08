"""Tests for pipeline.experiment_tracking: SQLite run log."""

import pytest

from pipeline.experiment_tracking import (
    SCHEMA_VERSION,
    connect,
    export_jsonl,
    list_runs,
    log_run,
    show_run,
)


def _log_minimal(conn, run_id="abc123"):
    return log_run(
        conn,
        backend="rule",
        backend_version="1.0",
        model_id="rule-keyword-scorer/1.0",
        prompt_path="pipeline/prompts/v1_classify.md",
        prompt_sha256="a" * 64,
        batch_id="20261008T000000Z-deadbeef",
        n_items=12,
        labels=["diabetes", "hypertension"],
        config={"chunk_tokens": 256},
        metrics={"accuracy_vs_ref": (0.9, 0.7, 0.98)},
        seed=42,
        run_id=run_id,
    )


def test_log_and_show_round_trip(tmp_path):
    conn = connect(str(tmp_path / "exp.db"))
    run_id = _log_minimal(conn)
    run = show_run(conn, run_id)
    assert run["backend"] == "rule"
    assert run["model_id"] == "rule-keyword-scorer/1.0"
    assert run["prompt_sha256"] == "a" * 64
    assert run["labels"] == ["diabetes", "hypertension"]
    assert run["config"] == {"chunk_tokens": 256}
    assert run["schema_version"] == SCHEMA_VERSION
    assert run["metrics"][0]["name"] == "accuracy_vs_ref"
    assert run["metrics"][0]["value"] == pytest.approx(0.9)


def test_list_runs_order(tmp_path):
    conn = connect(str(tmp_path / "exp.db"))
    _log_minimal(conn, run_id="first")
    _log_minimal(conn, run_id="second")
    runs = list_runs(conn)
    assert [r["run_id"] for r in runs] == ["second", "first"]  # newest first


def test_export_jsonl(tmp_path):
    conn = connect(str(tmp_path / "exp.db"))
    run_id = _log_minimal(conn)
    out = str(tmp_path / "run.jsonl")
    export_jsonl(conn, run_id, out)
    import json

    with open(out, encoding="utf-8") as f:
        data = json.loads(f.read())
    assert data["run_id"] == run_id


def test_show_missing_run_raises(tmp_path):
    conn = connect(str(tmp_path / "exp.db"))
    with pytest.raises(KeyError):
        show_run(conn, "nope")


def test_log_rejects_bad_prompt_hash(tmp_path):
    conn = connect(str(tmp_path / "exp.db"))
    with pytest.raises(ValueError):
        log_run(
            conn,
            backend="rule",
            backend_version="1.0",
            model_id="m",
            prompt_path="p",
            prompt_sha256="tooshort",
            batch_id="b",
            n_items=1,
            labels=["x"],
            config={},
            metrics={"m": (1.0, None, None)},
        )


def test_schema_version_stored(tmp_path):
    conn = connect(str(tmp_path / "exp.db"))
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert int(row[0]) == SCHEMA_VERSION
