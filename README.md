# Can an LLM replace human annotators?

A reliability study that treats Gemini 3.5 Flash-Lite like a new human annotator
and runs it through the standard reliability checks: accuracy with confidence
intervals, chance-corrected agreement, calibration of its confidence scores, a
paired comparison against a small supervised baseline, error analysis, and a
measured cost comparison.

## Bottom line

On 248 PubMed abstracts labeled with four disease topics, the LLM reached
**96.0% accuracy (95% CI 92.7% to 97.8%)**, **Cohen's kappa 0.946
(95% CI 0.909 to 0.978)** and **Gwet's AC1 0.946 (95% CI 0.909 to 0.979)**.
Its confidence scores are well calibrated (expected calibration error 0.017),
and it cost **$0.21 per 1,000 annotations** measured from exact token counts,
against an estimated $250 per 1,000 for human labeling. McNemar's exact test
found no detectable difference between the zero-shot LLM and a TF-IDF +
logistic regression model trained on labeled data (p = 0.508, 124 shared test
items: baseline 94.4%, LLM 96.8%).

The honest verdict: a strong, very cheap first-pass annotator for this kind
of topic labeling, but not a replacement for human adjudication where labels
carry real consequences. The practical pattern is LLM labels everything,
humans review the low-confidence and disputed items.

## Data

248 abstracts fetched from PubMed via the Entrez API (October 2026) using four
fixed MeSH queries: "diabetes mellitus", "hypertension", "asthma", "migraine
disorders", 62 abstracts per topic. Each abstract's label is the query that
retrieved it. This is **distant supervision, not a hand-verified gold
standard**: an article can mention a disease without being mainly about it, so
some "errors" are disagreements with a noisy label rather than genuine LLM
mistakes. The agreement numbers are best read as agreement with the topic
search's own judgment, and as an upper bound on agreement with careful human
annotation. The four topics also have distinctive vocabularies, so this is an
easier-than-average annotation task.

`data/build_dataset.py` regenerates the dataset (fixed seed 20261002).

## Methods

- **Annotation protocol:** one API call per abstract, Gemini 3.5 Flash-Lite,
  temperature 0, strict JSON output (`label` + `confidence` in [0,1]), 1s
  spacing between calls, retries with backoff. Every raw response cached to
  `data/api_cache/` before parsing; defensive parsing with parse failures
  logged (0 of 248 in the final run after two rate-limit retries).
- **Accuracy** with the Wilson score 95% interval (better than Wald near 0/1).
- **Cohen's kappa** and **Gwet's AC1**, each with a 2,000-resample percentile
  bootstrap 95% CI. AC1 guards against the kappa paradox under uneven
  marginals.
- **Calibration:** confidences binned into deciles; expected calibration error
  and calibration curve.
- **Baseline:** TF-IDF + logistic regression trained on a labeled 50% split,
  evaluated on the same 124 items the LLM labeled. Compared with McNemar's
  exact test on paired predictions. Note the asymmetry: the baseline saw
  labeled training data, the LLM was zero-shot.

## Key results

| Metric | Estimate | 95% CI |
|---|---|---|
| Accuracy | 0.960 | 0.927 to 0.978 (Wilson) |
| Cohen's kappa | 0.946 | 0.909 to 0.978 (bootstrap) |
| Gwet's AC1 | 0.946 | 0.909 to 0.979 (bootstrap) |
| Expected calibration error | 0.017 | -- |

Per-class: diabetes precision 0.871 / recall 0.984; hypertension precision
0.981 / recall 0.855; asthma and migraine 1.000 across the board. The main
error pattern is hypertension abstracts misclassified as diabetes (9 of the
10 total errors). Accuracy is flat across abstract-length quartiles
(0.935 to 0.984, overlapping CIs).

Cost (measured): 126,473 input tokens + 5,237 output tokens at the published
Gemini 3.5 Flash-Lite price ($0.30 / 1M input, $2.50 / 1M output, verified
October 2026) = $0.051 for all 248 annotations, i.e. $0.21 per 1,000.
Human figure is an estimate: published simple biomedical labeling rates run
about $0.10-$0.50 per item; $0.25 per item used here ($250 per 1,000),
excluding recruitment, training, and QC overhead.

## Limitations and threats to validity

1. Distant labels from MeSH queries, not adjudicated human annotation.
2. Four well-separated topics; results will not transfer to fine-grained or
   overlapping label sets.
3. One model, one prompt, temperature 0; no prompt sensitivity analysis.
4. Abstracts shorter than 300 characters were excluded at build time.
5. LLM cost is measured; human cost is an estimate.
6. One annotation pass per item; test-retest stability not measured.

Full discussion in REPORT.md.

## How to run

Requires Python 3.12 with numpy, pandas, scikit-learn, scipy, matplotlib,
plotly, nbformat, nbclient.

```bash
cd llm-annotator-reliability
python3 data/build_dataset.py     # rebuild the dataset (hits Entrez; optional)
python3 data/annotate.py         # annotate via the Gemini API (needs credential)
python3 src/run_analysis.py      # all statistics, figures/, results.json
python3 notebooks/build_notebook.py
python3 - <<'EOF'                # execute the notebook
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("notebooks/analysis.ipynb", as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
nbformat.write(nb, "notebooks/analysis.ipynb")
EOF
python3 dashboard_build.py       # rebuild dashboard.html
python3 report_build.py          # rebuild REPORT.md
```

The Gemini calls use the stored `custom.google-gemini` connector credential
via `src/gemini_client.py` (surrogate auth, requests restricted to
generativelanguage.googleapis.com). No keys are stored in the repo; grep for
keys before pushing is still recommended.

## Repo structure

```
llm-annotator-reliability/
  README.md               this file
  REPORT.md               full write-up with all tables and interpretation
  results.json            every computed number, machine-readable
  dashboard.html          self-contained Plotly dashboard (Overview, Methods,
                          Results, Limitations)
  src/
    gemini_client.py      Gemini API client (surrogate auth, caching, retries)
    metrics.py            wilson_ci, kappa, AC1, bootstrap CIs, McNemar, ECE
    plots.py              publication figures (matplotlib)
    run_analysis.py       full analysis -> results.json + figures/
  notebooks/
    analysis.ipynb        executed notebook with narrative (build via
    build_notebook.py       build_notebook.py, then execute)
  data/
    abstracts.csv         the 248 labeled abstracts
    annotations.csv       LLM labels + confidences + token counts
    api_cache/            raw API responses, one file per abstract
    build_dataset.py      Entrez fetch + stratified sampling (seed 20261002)
    annotate.py           annotation run
  figures/
    confusion_matrix.png  calibration.png  per_class_metrics.png
    cost_comparison.png
```

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales.
  Educational and Psychological Measurement, 20(1), 37-46.
- Gwet, K. L. (2008). Computing inter-rater reliability and its variance in
  the presence of high agreement. British Journal of Mathematical and
  Statistical Psychology, 61(1), 29-48.
- Wilson, E. B. (1927). Probable inference, the law of succession, and
  statistical inference. JASA, 22(158), 209-212.
- Guo, C. et al. (2017). On calibration of modern neural networks. ICML.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing
  supervised classification learning algorithms. Neural Computation, 10(7).
