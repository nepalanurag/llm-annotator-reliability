# Can an LLM replace human annotators? Full report

**Author:** Anurag Nepal, October 2026
**Model under test:** Gemini 3.5 Flash-Lite (`models/gemini-3.5-flash-lite`), temperature 0
**Items:** 248 PubMed abstracts, four disease topics
**Reproducibility:** fixed seed 20261002; raw API responses cached in `data/api_cache/`

## 1. Background and research question

Annotated text is the raw material of biomedical NLP: topic labels feed
classifiers, information retrieval, and systematic reviews. Human annotation is
slow and expensive, so the field keeps asking whether a large language model
can do the labeling instead. The question is rarely answered with proper
statistics. Accuracy alone is quoted without intervals, agreement is not
chance-corrected, and confidence scores are taken at face value.

This study treats the LLM exactly like a new human annotator joining a
labeling team and runs it through the standard reliability battery:
accuracy with a confidence interval, chance-corrected agreement (Cohen's
kappa and Gwet's AC1, each with bootstrap intervals), per-class diagnostics,
calibration of the model's own confidence scores, a comparison against a
small supervised baseline with a paired test, an error analysis, and a
measured cost comparison. The verdict is deliberately qualified: the goal is
to say *where* the LLM is reliable enough, not to declare victory or defeat.

## 2. Data

248 abstracts were fetched from PubMed through the Entrez API in October 2026
using four fixed MeSH queries: "diabetes mellitus", "hypertension", "asthma",
and "migraine disorders". Each abstract keeps a title and abstract text
(median 265 words). The label is the query that retrieved the article: 62
abstracts per topic.

This is distant supervision, not a hand-annotated gold standard, and that
shapes everything downstream. A "true" label here means "the topic search
found this article under this MeSH term". Articles that merely mention a
disease in passing can carry that disease's label, and comorbidity is common
in biomedical text. Consequences:

1. Disagreements between the LLM and the label are not always LLM errors;
   some are label noise. The agreement numbers are best read as agreement
   with the search query's judgment.
2. Because the four topics have fairly distinctive vocabularies, this is an
   easier-than-average annotation task. Results will not transfer directly to
   fine-grained or overlapping label sets.

The build is fully reproducible: `data/build_dataset.py` runs the fixed
queries with a fixed seed (see `data/abstracts.csv`).

## 3. Methods

### 3.1 Annotation protocol

One API call per abstract. Prompt: classify into exactly one of the four
topic labels and return a confidence in [0,1] as strict JSON. Settings:
temperature 0 (deterministic decoding), `responseMimeType: application/json`,
max 64 output tokens. Calls were spaced at least one second apart, with
exponential-backoff retries on network errors. Every raw response was cached
to `data/api_cache/` before parsing, so any single item can be audited.
Parsing was defensive: outputs that were not valid JSON, had a label outside
the set, or a confidence outside [0,1] were logged as parse failures rather
than coerced. 0 of 248 items failed this way.

### 3.2 Accuracy and uncertainty

Accuracy is reported with the Wilson score 95% interval. The common Wald
interval (p +/- 1.96*sqrt(p(1-p)/n)) undercovers when the proportion is near
0 or 1; our accuracy is high, so Wilson is the honest choice.

### 3.3 Chance-corrected agreement

Cohen's kappa = (observed agreement - chance agreement) / (1 - chance
agreement), where chance agreement is computed from the two raters' marginal
label frequencies. Kappa is the field standard but has a known paradox: with
very high agreement and uneven marginals it can look much worse than the raw
agreement suggests. We therefore also compute Gwet's AC1, which estimates
chance agreement from the average category proportions and is stable in
exactly that regime (Gwet 2008). For both statistics we use the percentile
bootstrap with 2,000 resamples (fixed seed): resample items with replacement,
recompute, take the 2.5th and 97.5th percentiles. The bootstrap assumes
nothing about the sampling distribution's shape, which matters for ratio
statistics like kappa.

### 3.4 Calibration

The model reports a confidence per label. Calibration asks whether a 0.9
means "right 90% of the time". We bin confidences into deciles, plot observed
accuracy against mean confidence in each bin, and summarize with the expected
calibration error: ECE = sum over bins of bin_weight * |accuracy -
mean_confidence|. A calibrated model sits on the diagonal; systematic gaps
mean over- or under-confidence. This matters operationally: a common
deployment pattern is "let the LLM label everything, send low-confidence
items to humans", and that pattern only works if the confidence is
calibrated.

### 3.5 Baseline comparison

We trained a TF-IDF + logistic regression classifier on a labeled 50% split
(stratified) and evaluated both it and the LLM on the remaining 124
items. Read this comparison with its asymmetry in mind: the baseline saw
labeled training data and the LLM saw none. It is zero-shot LLM versus a
small supervised model, not a fair fight. The practical question it answers:
if you have a little labeled data, is training something worth it?

Because both models are scored on the same items, predictions are paired. We
use McNemar's exact test, which looks only at discordant pairs (items where
exactly one model is right) and asks whether the split is consistent with a
coin flip. The exact binomial p-value avoids the chi-square approximation,
which is poor when discordant counts are small.

### 3.6 Error analysis and cost

Accuracy is sliced by abstract length quartile and by topic, each with Wilson
intervals. Cost uses the exact per-call token counts returned by the API
(`usageMetadata`) times the published Gemini 3.5 Flash-Lite list price
($0.30 per 1M input tokens, $2.50 per 1M output tokens, verified October
2026). The human figure is an estimate: published rates for simple biomedical
labeling cluster around $0.10-$0.50 per item, so $0.25 per item is used and
labeled as an estimate throughout.

## 4. Results

### 4.1 Overall accuracy and agreement

| Metric | Estimate | 95% CI |
|---|---|---|
| Accuracy | 0.960 | 0.927 to 0.978 (Wilson) |
| Cohen's kappa | 0.946 | 0.909 to 0.978 (bootstrap) |
| Gwet's AC1 | 0.946 | 0.909 to 0.979 (bootstrap) |
| Expected calibration error | 0.017 | -- |

The LLM agreed with the PubMed topic labels on 96.0% of items
(95% CI 92.7% to 97.8%). Kappa and AC1 both sit in the range
conventionally read as strong agreement, and their intervals exclude anything
near chance. AC1 running close to kappa is reassuring: it means the kappa
value is not an artifact of the kappa paradox here.

### 4.2 Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| diabetes | 0.871 | 0.984 | 0.924 | 62 |
| hypertension | 0.981 | 0.855 | 0.914 | 62 |
| asthma | 1.000 | 1.000 | 1.000 | 62 |
| migraine | 1.000 | 1.000 | 1.000 | 62 |

Precision answers "when the LLM says asthma, how often is it right";
recall answers "of the true asthma abstracts, how many did it find". The
confusion matrix (`figures/confusion_matrix.png`) shows where the residual
errors concentrate.

### 4.3 Calibration

| Confidence bin | n | Mean confidence | Observed accuracy |
|---|---|---|---|
| 0.3-0.4 | 7 | 0.400 | 0.571 |
| 0.4-0.5 | 7 | 0.479 | 0.429 |
| 0.5-0.6 | 3 | 0.567 | 0.333 |
| 0.6-0.7 | 1 | 0.650 | 1.000 |
| 0.8-0.9 | 6 | 0.850 | 1.000 |
| 0.9-1.0 | 224 | 0.992 | 0.996 |

ECE = 0.017. See `figures/calibration.png` for the curve. The practical
read: the bins where confidence runs ahead of accuracy are the bins where a
human-review threshold should be set conservatively.

### 4.4 Baseline comparison

On the 124 shared test items, the TF-IDF + logistic regression
baseline reached 0.944 accuracy; the LLM reached
0.968 on the same items. Discordant pairs:
3 where only the baseline was right, 6
where only the LLM was right. McNemar exact p = 0.5078.

Interpretation, stated carefully: this does not crown a winner. The baseline
had labeled training data; the LLM had none. What the test tells us is
whether a cheap supervised model trained on half the data is detectably
different from zero-shot LLM labeling on this task.

### 4.5 Error analysis

Accuracy by abstract length quartile (Wilson 95% CIs):

| Length quartile | Accuracy | 95% CI | n |
|---|---|---|---|
| Q1 (shortest) | 0.968 | 0.890 to 0.991 | 62 |
| Q2 | 0.935 | 0.846 to 0.975 | 62 |
| Q3 | 0.984 | 0.917 to 0.997 | 64 |
| Q4 (longest) | 0.950 | 0.863 to 0.983 | 60 |

Accuracy by topic:

| Topic | Accuracy | n |
|---|---|---|
| asthma | 1.000 | 62 |
| diabetes | 0.984 | 62 |
| hypertension | 0.855 | 62 |
| migraine | 1.000 | 62 |

### 4.6 Cost

Total tokens used: 126,473 input,
5,237 output. At list price that is
$0.0510 for all 248 annotations, i.e.
**$0.21 per 1,000 annotations (measured)**.
The human estimate is **$250 per
1,000 annotations** at $0.25/item, excluding recruitment, training, and
quality-control overhead, which would widen the gap further.

## 5. Interpretation

Three findings matter for practice.

First, agreement is strong but not perfect. 96.0% accuracy with a lower
confidence bound of 92.7% means roughly one label in 14 to
25 is wrong even in this favorable setting. That error rate is fine
for exploratory labeling and for generating training data that will be
checked, and it is too high for any pipeline where a single mislabel is
expensive.

Second, the confidence scores carry real signal (ECE 0.017), which makes
the hybrid pattern viable: LLM labels everything, humans review the
low-confidence and disputed items. The calibration curve, not the aggregate
ECE, should set the review threshold.

Third, the economics are lopsided. Measured LLM cost is two orders of
magnitude below the human estimate. The expensive part of an LLM annotation
pipeline is not the API bill; it is the human review time and the design of
the adjudication step.

## 6. Limitations and threats to validity

1. **Distant labels.** The reference labels come from MeSH queries, not
   adjudicated human annotation. True agreement with careful human labels is
   likely lower.
2. **Easy label set.** Four diseases with distinctive vocabularies. Do not
   generalize to fine-grained or overlapping categories.
3. **Single model, single prompt.** Results describe Gemini 3.5 Flash-Lite
   with this exact prompt at temperature 0. Prompt wording can move accuracy
   by several points; no prompt sensitivity analysis was run.
4. **Clean inputs.** Abstracts shorter than 300 characters were excluded at
   build time. Real pipelines include messy inputs.
5. **Cost asymmetry.** The LLM cost is measured from exact token counts; the
   human cost is an estimate that excludes overhead on the human side.
6. **No repeated runs.** One annotation pass per item. LLM outputs can vary
   across runs even at temperature 0; test-retest stability was not measured.

## 7. Reproducibility

- Fixed seed 20261002 for dataset sampling, train/test split, and bootstrap.
- `data/build_dataset.py` regenerates the dataset; `data/annotate.py`
  re-annotates (raw responses cached in `data/api_cache/`).
- `src/run_analysis.py` recomputes every number, figure, and `results.json`.
- `notebooks/build_notebook.py` + `jupyter nbconvert --execute` rebuild the
  executed notebook; `dashboard_build.py` rebuilds `dashboard.html`.

## References

- Brown, T. et al. (2020). Language models are few-shot learners. NeurIPS 33.
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
