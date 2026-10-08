"""Dagster pipeline: extract -> chunk -> dedup -> annotate -> agreement.

Assets:
    extract_abstracts  arXiv API (or local CSV) -> RawAbstract contract
    chunk_abstracts    sentence-aware chunks + boundary checks
    deduplicate_chunks exact + MinHash dedup, stats in asset metadata
    annotate_batch     backend annotation -> versioned batch dir + manifest
    agreement_metrics  vs distant reference labels: accuracy, Cohen kappa,
                       Krippendorff alpha, per-category agreement
    versioned_results  metrics.json into the batch dir; returns batch_id

Jobs:
    annotation_job     the full batch pipeline
    drift_job          weekly label-drift check vs the committed baseline

Run locally without a daemon:
    dagster job execute ...  (or execute_in_process in tests)

Note: this module intentionally does NOT use `from __future__ import
annotations` — Dagster 1.13 cannot resolve PEP 563 string annotations on
pythonic Config classes.
"""

import json
import os

import pandas as pd
from dagster import (
    AssetExecutionContext,
    Config,
    Definitions,
    asset,
    define_asset_job,
    job,
    op,
    schedule,
)

from pipeline import CONTRACT_VERSION
from pipeline.annotate import run_batch
from pipeline.chunking import Chunk, assert_boundaries_ok, chunk_text
from pipeline.config import AnnotSettings
from pipeline.deduplication import deduplicate
from pipeline.experiment_tracking import connect, log_run
from pipeline.extract import extract_arxiv, extract_csv
from pipeline.logging import configure_logging, get_logger
from pipeline.metrics import (
    count_matrix_from_labels,
    fleiss_kappa,
    krippendorff_alpha,
    per_category_agreement,
    wilson_ci,
)
from pipeline.monitoring import check_drift
from pipeline.schemas import validate_or_raise, AgreementMetrics, Chunk as ChunkSchema

log = get_logger(__name__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _settings() -> AnnotSettings:
    return AnnotSettings()


class ExtractConfig(Config):
    source: str = "csv"  # "csv" | "arxiv"
    csv_path: str = "data/abstracts.csv"
    query: str = ""
    max_results: int = 50
    limit: int = 0  # 0 = no limit; for small dry-run batches


class ChunkConfig(Config):
    chunk_tokens: int = 256
    overlap: float = 0.15


class AnnotateConfig(Config):
    backend: str = "rule"
    prompt_version: str = "v1"
    labels: str = "diabetes,hypertension,asthma,migraine"
    batch_dir: str = "pipeline/batches"


@asset
def extract_abstracts(
    context: AssetExecutionContext, config: ExtractConfig
) -> pd.DataFrame:
    """Extract raw abstracts; validate the RawAbstract contract."""
    settings = _settings()
    configure_logging(settings.log_format, settings.log_level)
    if config.source == "csv":
        path = config.csv_path
        if not os.path.isabs(path):
            path = os.path.join(REPO_ROOT, path)
        df = extract_csv(path)
    elif config.source == "arxiv":
        df = extract_arxiv(
            config.query or settings.arxiv_query,
            max_results=config.max_results,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.max_retries,
            polite_delay_s=settings.polite_delay_s,
        )
    else:
        raise ValueError(f"source must be 'csv' or 'arxiv', got {config.source!r}")
    if config.limit and len(df) > config.limit:
        # Deterministic head for dry runs (no silent sampling).
        df = df.head(config.limit).reset_index(drop=True)
    context.add_output_metadata({"n_abstracts": len(df), "source": config.source})
    log.info("asset_extract_abstracts", n=len(df))
    return df


@asset(deps=[extract_abstracts])
def chunk_abstracts(
    context: AssetExecutionContext, config: ChunkConfig, extract_abstracts: pd.DataFrame
) -> pd.DataFrame:
    """Chunk abstracts; every chunk passes boundary sanity checks."""
    settings = _settings()
    chunks: list[Chunk] = []
    for _, row in extract_abstracts.iterrows():
        chunks.extend(
            chunk_text(
                doc_id=str(row["doc_id"]),
                text=str(row["text"]),
                chunk_tokens=config.chunk_tokens,
                overlap=config.overlap,
                min_chunk_chars=settings.min_chunk_chars,
                ref_label=(
                    str(row["ref_label"]) if pd.notna(row.get("ref_label")) else None
                ),
            )
        )
    assert_boundaries_ok(chunks, min_chunk_chars=settings.min_chunk_chars)
    df = pd.DataFrame(
        [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "n_chars": c.n_chars,
                "boundary_ok": True,
                "ref_label": c.ref_label,
            }
            for c in chunks
        ]
    )
    df = validate_or_raise(ChunkSchema, df)
    context.add_output_metadata({"n_chunks": len(df)})
    log.info("asset_chunk_abstracts", n_chunks=len(df))
    return df


@asset(deps=[chunk_abstracts])
def deduplicate_chunks(
    context: AssetExecutionContext, chunk_abstracts: pd.DataFrame
) -> pd.DataFrame:
    """Drop exact + near-duplicate chunks before spending annotation budget."""
    settings = _settings()
    chunks = [
        Chunk(
            r.doc_id,
            r.chunk_id,
            int(r.chunk_index),
            r.text,
            r.ref_label if pd.notna(r.ref_label) else None,
        )
        for r in chunk_abstracts.itertuples()
    ]
    result = deduplicate(
        chunks,
        threshold=settings.dedup_threshold,
        num_perm=settings.minhash_permutations,
        min_chunk_chars=settings.min_chunk_chars,
    )
    stats = result.stats()
    context.add_output_metadata(
        {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
    )
    df = pd.DataFrame(
        [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "n_chars": c.n_chars,
                "boundary_ok": True,
                "ref_label": c.ref_label,
            }
            for c in result.kept
        ]
    )
    df = validate_or_raise(ChunkSchema, df)
    log.info("asset_deduplicate_chunks", **stats)
    return df


@asset(deps=[deduplicate_chunks])
def annotate_batch(
    context: AssetExecutionContext,
    config: AnnotateConfig,
    deduplicate_chunks: pd.DataFrame,
) -> dict:
    """Annotate kept chunks; write the versioned batch dir; return manifest."""
    chunks = [
        Chunk(
            r.doc_id,
            r.chunk_id,
            int(r.chunk_index),
            r.text,
            r.ref_label if pd.notna(r.ref_label) else None,
        )
        for r in deduplicate_chunks.itertuples()
    ]
    labels = [label.strip() for label in config.labels.split(",") if label.strip()]
    batch_dir = config.batch_dir
    if not os.path.isabs(batch_dir):
        batch_dir = os.path.join(REPO_ROOT, batch_dir)
    manifest = run_batch(
        chunks,
        labels,
        config.backend,
        config.prompt_version,
        batch_id=_batch_id(),
        batch_dir=batch_dir,
    )
    # Persist the reference labels alongside for the agreement stage.
    ref = deduplicate_chunks[["chunk_id", "ref_label"]].copy()
    ref.to_parquet(
        os.path.join(batch_dir, manifest["batch_id"], "ref_labels.parquet"), index=False
    )
    context.add_output_metadata(
        {
            "batch_id": manifest["batch_id"],
            "n_annotated": manifest["n_annotated"],
            "backend": manifest["backend"],
        }
    )
    return manifest


def _batch_id() -> str:
    import datetime as dt
    import uuid

    return (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )


def compute_agreement(
    annotations: pd.DataFrame, ref_labels: pd.DataFrame, batch_id: str
) -> pd.DataFrame:
    """Agreement of backend labels vs distant reference labels.

    Metrics: accuracy + Wilson CI, Cohen's kappa (via sklearn), Krippendorff's
    alpha (nominal), per-category agreement (2-rater Fleiss form).
    """
    from sklearn.metrics import cohen_kappa_score

    merged = annotations.merge(ref_labels, on="chunk_id", how="left")
    merged = merged[merged["parsed_ok"] & merged["ref_label"].notna()].copy()
    if merged.empty:
        raise ValueError("no annotated chunks with reference labels for agreement")
    y_true = merged["ref_label"].astype(str).tolist()
    y_pred = merged["label"].astype(str).tolist()
    n = len(merged)
    categories = sorted(set(y_true) | set(y_pred))

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    lo, hi = wilson_ci(sum(1 for t, p in zip(y_true, y_pred) if t == p), n)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    alpha = krippendorff_alpha([[t, p] for t, p in zip(y_true, y_pred)])
    matrix = count_matrix_from_labels(
        [[t, p] for t, p in zip(y_true, y_pred)], categories
    )
    per_cat = per_category_agreement(matrix, categories)

    rows = [
        {
            "batch_id": batch_id,
            "metric": "accuracy_vs_ref",
            "value": acc,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": n,
        },
        {
            "batch_id": batch_id,
            "metric": "cohens_kappa_vs_ref",
            "value": kappa,
            "ci_lo": None,
            "ci_hi": None,
            "n": n,
        },
        {
            "batch_id": batch_id,
            "metric": "krippendorff_alpha_vs_ref",
            "value": alpha,
            "ci_lo": None,
            "ci_hi": None,
            "n": n,
        },
        {
            "batch_id": batch_id,
            "metric": "fleiss_kappa_2rater",
            "value": fleiss_kappa(matrix),
            "ci_lo": None,
            "ci_hi": None,
            "n": n,
        },
    ]
    for cat, val in per_cat.items():
        rows.append(
            {
                "batch_id": batch_id,
                "metric": f"per_category_agreement:{cat}",
                "value": float(val),
                "ci_lo": None,
                "ci_hi": None,
                "n": n,
            }
        )
    df = pd.DataFrame(rows)
    return validate_or_raise(AgreementMetrics, df)


@asset(deps=[annotate_batch])
def agreement_metrics(
    context: AssetExecutionContext, annotate_batch: dict
) -> pd.DataFrame:
    """Compute agreement metrics for the batch."""
    batch_dir = os.path.join(
        REPO_ROOT, "pipeline", "batches", annotate_batch["batch_id"]
    )
    annotations = pd.read_parquet(os.path.join(batch_dir, "annotations.parquet"))
    ref_labels = pd.read_parquet(os.path.join(batch_dir, "ref_labels.parquet"))
    df = compute_agreement(annotations, ref_labels, annotate_batch["batch_id"])
    for _, r in df.iterrows():
        context.add_output_metadata({str(r["metric"]): round(float(r["value"]), 4)})
    # Also log the run to the experiment tracker.
    settings = _settings()
    conn = connect(
        os.path.join(REPO_ROOT, settings.experiment_db)
        if not os.path.isabs(settings.experiment_db)
        else settings.experiment_db
    )
    metric_dict = {
        str(r["metric"]): (
            float(r["value"]),
            float(r["ci_lo"]) if pd.notna(r["ci_lo"]) else None,
            float(r["ci_hi"]) if pd.notna(r["ci_hi"]) else None,
        )
        for _, r in df.iterrows()
    }
    run_id = log_run(
        conn,
        backend=annotate_batch["backend"],
        backend_version=annotate_batch["backend_version"],
        model_id=annotate_batch["model_id"],
        prompt_path=f"pipeline/prompts/{annotate_batch['prompt_version']}_classify.md",
        prompt_sha256=annotate_batch["prompt_sha256"],
        batch_id=annotate_batch["batch_id"],
        n_items=annotate_batch["n_annotated"],
        labels=annotate_batch["labels"],
        config={"contract_version": CONTRACT_VERSION},
        metrics=metric_dict,
    )
    context.add_output_metadata({"experiment_run_id": run_id})
    log.info(
        "asset_agreement_metrics", batch_id=annotate_batch["batch_id"], run_id=run_id
    )
    return df


@asset(deps=[annotate_batch, agreement_metrics])
def versioned_results(
    context: AssetExecutionContext,
    annotate_batch: dict,
    agreement_metrics: pd.DataFrame,
) -> str:
    """Write metrics.json into the batch dir. Returns the batch_id."""
    batch_dir = os.path.join(
        REPO_ROOT, "pipeline", "batches", annotate_batch["batch_id"]
    )
    metrics_path = os.path.join(batch_dir, "metrics.json")
    agreement_metrics.to_json(metrics_path, orient="records", indent=2)
    context.add_output_metadata(
        {"batch_id": annotate_batch["batch_id"], "metrics_path": metrics_path}
    )
    log.info("asset_versioned_results", batch_id=annotate_batch["batch_id"])
    return annotate_batch["batch_id"]


annotation_job = define_asset_job(
    name="annotation_job",
    selection=[
        extract_abstracts,
        chunk_abstracts,
        deduplicate_chunks,
        annotate_batch,
        agreement_metrics,
        versioned_results,
    ],
)


@op
def drift_check_op(context: AssetExecutionContext) -> dict:
    """Weekly drift check: latest batch label marginals vs committed baseline."""
    settings = _settings()
    baseline_path = (
        settings.baseline_path
        if os.path.isabs(settings.baseline_path)
        else os.path.join(REPO_ROOT, settings.baseline_path)
    )
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"baseline not found: {baseline_path}. Create it from a trusted batch "
            "with pipeline/monitoring.py inputs."
        )
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    batch_root = (
        settings.batch_dir
        if os.path.isabs(settings.batch_dir)
        else os.path.join(REPO_ROOT, settings.batch_dir)
    )
    batches = (
        sorted(
            d
            for d in os.listdir(batch_root)
            if os.path.isdir(os.path.join(batch_root, d))
        )
        if os.path.isdir(batch_root)
        else []
    )
    if not batches:
        raise FileNotFoundError(f"no batches under {batch_root} to monitor")
    latest = os.path.join(batch_root, batches[-1], "annotations.parquet")
    df = pd.read_parquet(latest)
    incoming = df[df["parsed_ok"]]["label"].value_counts().to_dict()
    verdict = check_drift(
        baseline,
        {str(k): int(v) for k, v in incoming.items()},
        psi_alert=settings.psi_alert_threshold,
        psi_warn=settings.psi_warning_threshold,
        chi2_alpha=settings.chi2_alpha,
    )
    context.add_output_metadata({"verdict": verdict["verdict"], "psi": verdict["psi"]})
    if verdict["alert"]:
        log.warning(
            "drift_alert",
            **{k: v for k, v in verdict.items() if k not in ("baseline", "incoming")},
        )
    return verdict


@job
def drift_job():
    drift_check_op()


@schedule(
    cron_schedule="0 9 * * 1", job=drift_job, execution_timezone="America/Los_Angeles"
)
def weekly_drift_schedule(_context):
    """Mondays 09:00 PT: label-drift check on the latest annotation batch."""
    return {}


defs = Definitions(
    assets=[
        extract_abstracts,
        chunk_abstracts,
        deduplicate_chunks,
        annotate_batch,
        agreement_metrics,
        versioned_results,
    ],
    jobs=[annotation_job, drift_job],
    schedules=[weekly_drift_schedule],
)
