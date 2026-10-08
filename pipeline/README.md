# Annotation pipeline

Production annotation infrastructure for the llm-annotator-reliability
research: extract → chunk → dedup → annotate → agreement metrics → versioned
results, with experiment tracking, label-drift monitoring, and AI-assisted
adversarial analysis. Built 2026-10-08 as the data-engineering companion to
the published reliability study (see the repo root README).

Design rules: typed code throughout, structlog (never `print` for logging),
pydantic-settings config, pandera data contracts versioned with
`CONTRACT_VERSION`, pytest with hand-verified reference values, pinned deps,
pre-commit (black + ruff), Dockerfile, and GitHub Actions CI. Every CLI has
`--help`, sensible defaults, and fails loudly on bad input.

## Architecture

```
arXiv API ──► extract ──► chunk ──► dedup ──► annotate ──► agreement ──► versioned batch
(local CSV)      │            │          │           │             │
                 ▼            ▼          ▼           ▼             ▼
           RawAbstract    Chunk     dedup stats  Annotation   accuracy, Cohen κ,
           contract      contract   (MinHash)    contract     Fleiss κ, Krippendorff α
                                                              per-category agreement
```

Cross-cutting:

- **Experiment tracking** (`experiment_tracking.py`): every run logs backend,
  model id, prompt file + sha256, batch config, and metrics to SQLite
  (`pipeline/runs/experiments.db`, schema-v1) with JSONL export.
- **Monitoring** (`monitoring.py` + weekly Dagster schedule): PSI and
  chi-square drift tests of incoming label marginals vs the committed
  baseline; markdown drift reports; non-zero exit on ALERT for schedulers.
- **AI components**: `ai_adversarial.py` (hard-case generator targeting the
  weakest categories) and `ai_clustering.py` (unsupervised disagreement
  taxonomy vs the human-written error analysis).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r pipeline/requirements.txt   # pinned

# 1. Extract (offline path uses the repo's published corpus)
python -m pipeline.extract --from-csv data/abstracts.csv --out /tmp/raw.parquet

# 2. Chunk
python -m pipeline.chunking_cli /tmp/raw.parquet --out /tmp/chunks.jsonl

# 3. Dedup
python -m pipeline.deduplication /tmp/chunks.jsonl --out /tmp/kept.jsonl --stats /tmp/dedup.json

# 4. Annotate (rule backend: offline, $0)
python -m pipeline.annotate /tmp/kept.jsonl --backend rule --prompt v1 --batch-id demo

# 5. Full Dagster run (same stages, orchestrated; 12-abstract dry run)
python - <<'EOF'
from pipeline.defs import defs
job = defs.get_job_def("annotation_job")
result = job.execute_in_process(run_config={"ops": {
    "extract_abstracts": {"config": {"source": "csv",
                                     "csv_path": "data/abstracts.csv",
                                     "limit": 12}},
    "annotate_batch": {"config": {"backend": "rule", "prompt_version": "v1"}},
}})
assert result.success
EOF

# 6. Drift check on a new batch
python -m pipeline.monitoring --baseline pipeline/baselines/label_baseline.json \
    --incoming my_counts.json --report drift.md --fail-on-alert

# 7. Experiment log
python -m pipeline.experiment_tracking list
```

`dvc repro` runs stages extract → chunk → dedup → annotate as a plain
alternative to Dagster. `docker build -t annotator-pipeline .` for the image.

## Configuration

All settings are `ANNOT_*` env vars (see `pipeline/config.py`); `params.yaml`
holds the DVC-visible subset. Notable: `ANNOT_BACKEND` (rule|gemini),
`ANNOT_PROMPT_VERSION`, `ANNOT_DEDUP_THRESHOLD`, `ANNOT_PSI_ALERT_THRESHOLD`.

## Prompts are versioned files, never inline strings

`pipeline/prompts/v1_classify.md` ports the repo's published prompt verbatim;
`v2_classify.md` adds the comorbidity tie-break rule aimed at the
diabetes/hypertension boundary (the repo's 9-of-10 error cluster). The sha256
of the rendered prompt is stamped on every annotation and experiment-log row,
so any result is traceable to the exact prompt text.

## Backends

- `rule` — deterministic keyword scorer. Offline, $0, no credentials. A weak
  annotator on purpose: it exercises the whole pipeline for free and serves as
  the measuring stick. It abstains (rather than guessing) when no keyword hits.
- `gemini` — `models/gemini-3.5-flash-lite` with the versioned prompt, raw
  responses cached per item like the repo's client. Needs the
  `custom.google-gemini` connector credential at runtime; raises a clear error
  without it. Documented in `pipeline/backends.py`.

## Cost

Per-1k-annotation table with measured/derived/estimated labels:
[`pipeline/COST.md`](COST.md). Headline: rule $0.00 (~2 s/1k, measured);
Flash-Lite $0.21/1k (measured); human ~$250/1k (estimate).

## Human vs AI-assisted annotation workflow

This is the comparison the pipeline exists to make honest. Three workflows for
1,000 biomedical abstracts:

| Workflow | Cost / 1k | Time / 1k | Basis |
|---|---|---|---|
| Fully manual (human labels all) | ~$250 | ~8–17 h | ESTIMATE: repo's $0.25/item; 30–60 s/item |
| LLM first pass only (Flash-Lite) | $0.21 | ~52 min | MEASURED cost; time DERIVED from the 3 s/call politeness delay |
| **Pipeline-assisted (recommended)**: LLM labels all, human reviews low-confidence/disputed items only | **~$24.50** | **~0.8–1.6 h** | DERIVED from the repo's measured calibration bins (below) |

The assisted row is not hand-waving; it falls out of the repo's calibration
table (`results.json`): 224 of 248 items (90.3%) came back with confidence
≥ 0.9 at 0.9955 accuracy, so only ~97 of 1,000 items need human eyes.
97 × $0.25 + $0.21 ≈ **$24.50/1k** — roughly a **10× cost cut** and a
**10× time cut** versus fully manual, with the review threshold set by the
calibration curve (exactly what REPORT.md §5 recommends), not by the aggregate
ECE.

Agreement side, measured in this build (rule backend vs the repo's 248
published annotations):

- 3-rater Fleiss κ across [distant MeSH label, Gemini, rule baseline]: **0.924**
  (Krippendorff α identical at 0.924)
- Per-category agreement reproduces the repo's qualitative finding: the
  boundary categories are weakest (diabetes 0.820, hypertension 0.812) while
  asthma (0.956) and migraine (1.000) are near-unanimous
- Rule-vs-distant accuracy: 0.927 [0.888, 0.953] vs the LLM's 0.960 —
  the $0 baseline trails by ~3 points, which is the price of free

What is *not* claimed: a human-vs-hybrid agreement delta in κ points. The
repo has no human–human baseline (its labels are distant MeSH labels —
limitation #1), so quoting one would be fabrication. The measurable,
defensible delta is the review burden: **9.7% of items need adjudication**,
and the pipeline — calibration-gated review, drift alerts, adversarial
regression cases — is the infrastructure that keeps that 9.7% honest as
models and data drift.

## AI components

- **Adversarial cases** (`ai_adversarial.py`): generates deliberately hard
  cases for the six trap types aimed at the weakest categories
  (boundary comorbidity, drug-name traps, symptom overlap, multi-disease
  ambiguity, hedging, sarcastic reviews). `--dry-run` uses seeded templates
  (no API); `--llm` has Gemini author them. Runs the annotator, reports
  per-trap accuracy, writes `adversarial_report.md`. Dry-run finding on the
  rule backend: 0.692 overall; sarcastic reviews (0.333) and symptom overlap
  (0.500) break it, boundary/drug traps don't — those need `--llm` to measure
  the real annotator.
- **Disagreement taxonomy** (`ai_clustering.py`): TF-IDF + HDBSCAN over
  disagreements (repo's 10 real errors + seeded synthetic), rule-based cluster
  naming in `--dry-run`, Gemini naming in `--llm`. Compares the machine
  taxonomy against REPORT.md's human error analysis: ARI 0.184, mean cluster
  purity 1.0 — the machine finds *purer subtypes* of the human themes (finer
  granularity), a triage tool rather than a replacement for reading the
  confusion matrix. Writes `disagreement_taxonomy.md`.

## DVC and versioning

`dvc.yaml` defines extract → chunk → dedup → annotate. Annotation batches
(`pipeline/batches/<batch_id>/` with `manifest.json`, `annotations.parquet`,
`metrics.json`) are the DVC-tracked unit: `dvc add pipeline/batches/<id>`.
Batch dirs and the experiment DB are gitignored; the label baseline
(`pipeline/baselines/label_baseline.json`) is committed config.

## File map

```
pipeline/
  defs.py                 Dagster assets, annotation_job, weekly drift schedule
  extract.py              arXiv API / CSV extraction
  chunking.py             sentence-aware chunker + boundary checks
  chunking_cli.py         CLI wrapper for the chunker
  deduplication.py        exact + MinHash dedup, dedup stats
  backends.py             rule + gemini backends, versioned prompt loader
  annotate.py             batch annotation CLI → versioned batch dirs
  metrics.py              Fleiss κ, Krippendorff α, per-category agreement, Wilson
  experiment_tracking.py  SQLite run log + CLI
  monitoring.py           PSI + chi-square drift monitor + CLI
  ai_adversarial.py       hard-case generator + adversarial_report.md
  ai_clustering.py        disagreement taxonomy + disagreement_taxonomy.md
  schemas.py              pandera contracts (CONTRACT_VERSION 1.0)
  config.py               pydantic-settings (ANNOT_* env)
  logging.py              structlog setup
  prompts/v1_classify.md  published prompt, versioned
  prompts/v2_classify.md  + comorbidity tie-break rule
  tests/                  60 tests, hand-verified reference values
  baselines/              committed label baseline for drift monitoring
  batches/                runtime (gitignored, DVC-tracked)
  runs/                   experiment DB (gitignored)
```

## Changelog

- 2026-10-08: v0.1.0 — initial pipeline (CONTRACT_VERSION 1.0).
