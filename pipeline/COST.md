# Cost per 1,000 annotations

Token basis (measured, from the repo's `results.json`): 126,473 input +
5,237 output tokens for 248 annotations, i.e. **~510 input + ~21 output
tokens per annotation**, or **~510k input + ~21k output tokens per 1k**.

| Backend | $ / 1k annotations | Time / 1k | How the number was obtained |
|---|---|---|---|
| `rule` (keyword baseline) | **$0.00** | **~2 s** (measured: 632 items/s on this VM) | MEASURED 2026-10-08 |
| `gemini` / Flash-Lite | **$0.21** | ~52 min (derived) | MEASURED cost from exact token counts (`results.json`); time DERIVED from the repo client's 3 s politeness delay |
| `gemini` / Flash (full) | **~$0.95** | ~52 min (derived) | ESTIMATE: same token basis × list prices $1.50/$9.00 per 1M (verified Oct 2026) |
| Human annotator | **~$250** | ~8–17 h (estimate) | ESTIMATE: the repo's $0.25/item figure (REPORT.md 3.6; published biomedical labeling rates $0.10–$0.50); time assumes 30–60 s/item |

## Reading the table

- Every row is labeled MEASURED, DERIVED, or ESTIMATE. Measured beats derived
  beats estimate; the human row is the softest number here and is labeled as such.
- The expensive part of an LLM annotation pipeline is not the API bill — it is
  human review time and adjudication design (the repo's REPORT.md 5 makes the
  same point: $0.21 vs $250 is a ~1,200× gap on the API line alone).
- The `rule` backend exists to make the pipeline testable for $0: CI, dry runs,
  drift-monitor smoke tests, and adversarial baselines all run offline.
- Reproduce the rule-backend numbers: `python -m pipeline.annotate` on the
  repo's `data/abstracts.csv` (see README quickstart); throughput was timed
  with `time.perf_counter` around `RuleBackend.annotate` on 248 abstracts.
- Token basis will shift with prompt version (v2 is longer) and chunk size;
  the experiment log records exact per-run token counts for the `gemini`
  backend, so re-derive this table per prompt version before budgeting a run.
