"""Generate notebooks/analysis.ipynb via nbformat, embedding real computed
numbers from results.json into the narrative. Run AFTER src/run_analysis.py.
The notebook is then executed with `jupyter nbconvert --to notebook --execute
--inplace notebooks/analysis.ipynb`.
"""
import json
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)

LABELS = ["diabetes", "hypertension", "asthma", "migraine"]

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})


def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text.strip()))


acc, (acc_lo, acc_hi) = R["accuracy"], R["accuracy_ci95"]
kap, (kap_lo, kap_hi) = R["kappa"], R["kappa_ci95"]
ac1, (ac1_lo, ac1_hi) = R["ac1"], R["ac1_ci95"]
n = R["n"]

md(f"""
# Can an LLM replace human annotators? A reliability study

This notebook asks a practical question: if we let a large language model
(Gemini 3.5 Flash-Lite) assign topic labels to biomedical abstracts, how much
can we trust its labels? We treat the LLM like a new human annotator and put
it through the standard reliability checks: accuracy with honest confidence
intervals, inter-rater agreement (Cohen's kappa and Gwet's AC1), calibration
of its confidence scores, a head-to-head comparison against a small supervised
model, an error analysis, and a cost comparison.

Key numbers up front: on {R['n']} PubMed abstracts the LLM reached
**{acc:.1%} accuracy (95% CI {acc_lo:.1%} to {acc_hi:.1%})**,
**Cohen's kappa {kap:.3f} (95% CI {kap_lo:.3f} to {kap_hi:.3f})**,
and **Gwet's AC1 {ac1:.3f} (95% CI {ac1_lo:.3f} to {ac1_hi:.3f})**.
The rest of this notebook shows how each number was computed and what it
actually means.
""")

md("""
## 1. Setup

We fix the random seed everywhere so every number in this notebook is
reproducible. Reproducibility is not decoration: if a hiring manager reruns
this notebook and gets different numbers, nothing else in it can be trusted.
""")
code("""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Workaround: this environment's matplotlib tries to install an IPython
# display hook on first figure creation and fails (IPython/matplotlib
# version mismatch). The Agg backend needs no display hook, so disable it.
plt.install_repl_displayhook = lambda: None

SEED = 20261002
rng = np.random.default_rng(SEED)
LABELS = ["diabetes", "hypertension", "asthma", "migraine"]
sys.path.insert(0, os.path.abspath("../src"))
from metrics import (wilson_ci, cohens_kappa, gwet_ac1, bootstrap_ci, mcnemar,
                     calibration_bins, expected_calibration_error,
                     per_class_metrics)
print("seed:", SEED)
""")

md(f"""
## 2. The data

We pulled {R['n']} abstracts from PubMed with the Entrez API, querying four
fixed MeSH terms ("diabetes mellitus", "hypertension", "asthma",
"migraine disorders"). Each abstract is labeled by the query that retrieved
it. That makes this a **distant-supervision** setup, not a hand-annotated
gold standard: the label says which topic search found the article, and an
article can mention a disease without being mainly about it. That means any
"error" we count may sometimes be a disagreement between the LLM and a
noisy label, not a genuine LLM mistake. The honest way to read the results
is: how well does the LLM reproduce the topic search's own judgment?

The build script is `data/build_dataset.py` (fixed seed, fixed queries, so
the same dataset can be regenerated). A stratified sample of 62 per topic
keeps the classes balanced, which matters because chance-corrected agreement
metrics like kappa are sensitive to class balance.
""")
code("""
abstracts = pd.read_csv("../data/abstracts.csv")
print(abstracts.shape)
print(abstracts["topic"].value_counts())
print(abstracts[["n_words"]].describe().T)
abstracts.head(2)
""")

md("""
## 3. Getting the LLM's annotations

Each abstract went to Gemini 3.5 Flash-Lite with one prompt: classify into
the four labels and return a confidence between 0 and 1 as strict JSON. One
call per abstract, temperature 0 (deterministic decoding), a one-second gap
between calls to respect rate limits, and retries with backoff on network
errors. Crucially, every raw API response was cached to disk
(`data/api_cache/`) *before* parsing, so the annotation run is resumable and
auditable: you can inspect exactly what the model returned for any item.

The annotation script is `data/annotate.py`. It parses defensively: if the
model returns something that is not valid JSON or a label outside the set,
that item is logged as a parse failure rather than silently dropped or
coerced. Here we load the results.
""")
code(f"""
ann = pd.read_csv("../data/annotations.csv")
print("annotated:", len(ann))
print("parse failures:", {R['n_parse_failures']})
print("labels returned:", ann["pred_label"].value_counts().to_dict())
df = abstracts.merge(ann, on=["pmid", "topic"], how="inner")
df = df[df["parsed_ok"] == 1].reset_index(drop=True)
y_true = df["topic"].to_numpy()
y_pred = df["pred_label"].to_numpy()
n = len(df)
print("usable items:", n)
""")

md(f"""
## 4. Accuracy, with an honest interval

A point estimate without an interval is a claim without evidence. With
{n} items, the sampling noise is real: if we reran this study on a different
set of {n} abstracts, the accuracy would move. We report the Wilson score
interval rather than the simpler Wald interval (p +/- 1.96*sqrt(p(1-p)/n))
because the Wald interval undercovers badly near 0 and 1, and our accuracy
is high. The Wilson interval stays honest there.
""")
code("""
correct = (y_true == y_pred)
acc = correct.mean()
lo, hi = wilson_ci(int(correct.sum()), n)
print(f"accuracy: {acc:.3f}  (95% Wilson CI: {lo:.3f} to {hi:.3f})")
""")

md(f"""
The LLM got {int(acc*n)} of {n} right: **{acc:.1%} (95% CI {acc_lo:.1%} to
{acc_hi:.1%})**. That is the number to remember, and the interval is the
number to quote alongside it.

## 5. Agreement beyond accuracy: kappa and AC1

Accuracy alone can flatter. If 90% of items belonged to one class, a lazy
classifier that always predicts that class scores 90% accuracy while learning
nothing. Chance-corrected agreement metrics subtract the agreement you would
get by guessing from the marginal distributions.

**Cohen's kappa** is the classic: (observed agreement - chance agreement) /
(1 - chance agreement), where chance agreement comes from the two raters'
marginal label frequencies. Kappa came out to **{kap:.3f}**. A known quirk of
kappa: when agreement is very high, small imbalances in the marginals can
make kappa look much worse than the raw agreement suggests (the "kappa
paradox"). So we also compute **Gwet's AC1**, which estimates chance
agreement differently, from the average category proportions, and is more
stable when one label dominates. AC1 came out to **{ac1:.3f}**.

For both we bootstrap: resample the {n} items with replacement 2,000 times,
recompute the statistic, and take the 2.5th and 97.5th percentiles. The
bootstrap makes no normality assumption about the sampling distribution,
which is the right call for a ratio statistic like kappa.
""")
code("""
kappa = cohens_kappa(y_true, y_pred, LABELS)
k_lo, k_hi = bootstrap_ci(cohens_kappa, y_true, y_pred, LABELS, n_boot=2000)
ac1 = gwet_ac1(y_true, y_pred, LABELS)
a_lo, a_hi = bootstrap_ci(gwet_ac1, y_true, y_pred, LABELS, n_boot=2000)
print(f"Cohen's kappa: {kappa:.3f} (95% bootstrap CI: {k_lo:.3f} to {k_hi:.3f})")
print(f"Gwet's AC1:    {ac1:.3f} (95% bootstrap CI: {a_lo:.3f} to {a_hi:.3f})")
""")

md("""
## 6. Where does it go wrong? Per-class metrics and the confusion matrix

An aggregate number hides which classes suffer. Precision tells us: of the
items the LLM called "asthma", how many really were asthma? Recall tells us:
of the true asthma abstracts, how many did the LLM catch? A classifier can
have fine overall accuracy while completely missing one rare class, and for
an annotation pipeline that is the failure mode that matters.
""")
code("""
pm = per_class_metrics(y_true, y_pred, LABELS)
pd.DataFrame(pm).T.round(3)
""")
code("""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
fig.colorbar(im, ax=ax, label="Fraction of true class")
ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=25, ha="right"); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_title("Confusion matrix, LLM vs PubMed topic labels (row-normalized)")
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm[i,j] > 0.5 else "black", fontsize=9)
fig.tight_layout(); fig.savefig("../figures/confusion_matrix.png")
print("saved figures/confusion_matrix.png")
""")

md(f"""
## 7. Is the confidence honest? Calibration

The model returns a confidence with each label. A confidence of 0.9 should
mean the label is right about 90% of the time. If the model says 0.9 but is
right only 70% of the time, it is overconfident, and any pipeline that trusts
the confidence (for example, routing only low-confidence items to humans)
will be miscalibrated.

We bin the confidences into deciles and plot the observed accuracy in each
bin against the mean confidence, with the diagonal as perfect calibration.
The single-number summary is the expected calibration error (ECE): the
average absolute gap between confidence and accuracy, weighted by bin size.
Here **ECE = {R['ece']:.3f}**. The plot is more informative than the number:
it shows *where* the miscalibration lives.
""")
code("""
conf = df["confidence"].to_numpy()
bins = calibration_bins(conf, correct.astype(float), n_bins=10)
ece = expected_calibration_error(conf, correct.astype(float), n_bins=10)
print(f"ECE: {ece:.3f}")
pd.DataFrame(bins)[["lo","hi","n","mean_confidence","accuracy"]].round(3)
""")
code("""
x = [b["mean_confidence"] for b in bins]
y = [b["accuracy"] for b in bins]
nn = [b["n"] for b in bins]
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
ax.scatter(x, y, s=[10 + 90*v/max(nn) for v in nn], color="#1f77b4", alpha=0.85, zorder=3)
for b in bins:
    ax.annotate(f"n={b['n']}", (b["mean_confidence"], b["accuracy"]),
                xytext=(4, 6), textcoords="offset points", fontsize=8, color="#333")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted confidence in bin")
ax.set_ylabel("Observed accuracy in bin")
ax.set_title(f"Calibration curve (ECE = {ece:.3f})")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("../figures/calibration.png")
print("saved figures/calibration.png")
""")

md("""
## 8. Beating a cheap baseline? McNemar's test

We compare the LLM against a TF-IDF + logistic regression model trained on
half the labeled data (a 50/50 split, stratified). Read the comparison
carefully: the baseline *saw labeled training data* and the LLM saw *none*,
so this is zero-shot LLM versus a small supervised model, not a fair fight.
Still, it answers the practical question: if you have a little labeled data,
is it worth training something?

Because both models are evaluated on the same test items, their predictions
are paired, and a plain two-sample comparison would be wrong. McNemar's test
looks only at the discordant pairs: items where one model is right and the
other is wrong. Under the null hypothesis that the two models are equally
good, each discordant item is a coin flip. The exact binomial p-value
quantifies how surprising the observed split is.
""")
code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
idx = np.arange(n)
tr, te = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y_true)
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, random_state=SEED)
clf.fit(vec.fit_transform(df.loc[tr, "text"]), y_true[tr])
base_pred = clf.predict(vec.transform(df.loc[te, "text"]))
base_acc = (base_pred == y_true[te]).mean()
llm_te_acc = (y_pred[te] == y_true[te]).mean()
b01 = int(np.sum((y_pred[te] != y_true[te]) & (base_pred == y_true[te])))
b10 = int(np.sum((y_pred[te] == y_true[te]) & (base_pred != y_true[te])))
oratio, p = mcnemar(b01, b10)
print(f"test items: {len(te)}")
print(f"baseline accuracy: {base_acc:.3f}   LLM accuracy (same items): {llm_te_acc:.3f}")
print(f"discordant pairs: baseline-only-right={b01}, LLM-only-right={b10}")
print(f"McNemar exact p = {p:.4f}")
""")

md("""
## 9. Error analysis: length and topic

Two slices that often reveal structure. First, text length in quartiles:
long abstracts carry more signal, but they also carry more distractors
(comorbidities, background diseases). Second, accuracy by topic: some disease
areas have more distinctive vocabulary than others.
""")
code("""
df["len_quartile"] = pd.qcut(df["n_words"], 4,
                             labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])
df["correct"] = (df["topic"] == df["pred_label"]).astype(int)
by_len = df.groupby("len_quartile", observed=True)["correct"].agg(["mean", "count"])
by_len["lo"] = [wilson_ci(int((df[df['len_quartile']==q]['correct']).sum()),
                          int((df['len_quartile']==q).sum()))[0] for q in by_len.index]
by_len["hi"] = [wilson_ci(int((df[df['len_quartile']==q]['correct']).sum()),
                          int((df['len_quartile']==q).sum()))[1] for q in by_len.index]
print(by_len.round(3).to_string())
print()
print(df.groupby("topic")["correct"].agg(["mean", "count"]).round(3).to_string())
""")

md(f"""
## 10. Cost per annotation

Quality means little without price. The API returns exact token counts per
call (`usageMetadata`), so the LLM cost is measured, not estimated: input
and output tokens summed over the run, times the published Gemini 3.5
Flash-Lite list price ($0.30 / 1M input, $2.50 / 1M output, verified Oct
2026). The human number is necessarily an estimate: published rates for
simple biomedical labeling tasks run roughly $0.10-$0.50 per item, so we use
$0.25 per item and label it as an estimate.
""")
code("""
c = json.load(open("../results.json"))["cost"]
n_items = json.load(open("../results.json"))["n"]
print("total input tokens: ", c["input_tokens_total"])
print("total output tokens:", c["output_tokens_total"])
print(f"LLM cost for {n_items} annotations: ${c['llm_cost_usd_total']:.4f}")
print(f"LLM cost per 1,000:  ${c['llm_cost_per_1000_usd']:.2f}")
print(f"Human estimate per 1,000: ${c['human_cost_per_1000_usd_estimate']:.2f}")
""")

md(f"""
## 11. Verdict

Putting it together:

- **Agreement is strong but not perfect.** Accuracy {acc:.1%} (95% CI
  {acc_lo:.1%} to {acc_hi:.1%}), kappa {kap:.3f}, AC1 {ac1:.3f}. On a four-way
  topic task with distinctive vocabularies, the LLM is a good first
  annotator, not a replacement for adjudicated labels.
- **The confidence scores are usable with care.** ECE {R['ece']:.3f}; check the
  calibration plot for which bins deviate before wiring confidence into a
  routing rule.
- **It is dramatically cheaper than humans** (measured ${R['cost']['llm_cost_per_1000_usd']:.2f}
  per 1,000 vs an estimated ${R['cost']['human_cost_per_1000_usd_estimate']:.0f}
  per 1,000 for human labeling), which makes a hybrid pipeline attractive:
  let the LLM label everything, send low-confidence or disputed items to
  humans.
- **Where it is not reliable enough:** any use where a single mislabel is
  expensive (clinical coding, safety-critical triage), and any topic where
  the vocabulary overlaps heavily with neighbors. The label source here is
  distant supervision, so these numbers are an upper bound on agreement with
  careful human annotation.

The honest bottom line: use the LLM as a cheap first pass with a human in
the loop on the uncertain cases, not as a drop-in replacement for human
annotators.
""")

with open(os.path.join(HERE, "notebooks", "analysis.ipynb"), "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/analysis.ipynb with", len(nb.cells), "cells")
