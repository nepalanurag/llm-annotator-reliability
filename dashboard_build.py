"""Generate dashboard.html: self-contained Plotly dashboard (CDN), four
sections, all data embedded as JSON, figures embedded as base64 PNGs.
Run AFTER src/run_analysis.py.
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)

LABELS = ["diabetes", "hypertension", "asthma", "migraine"]
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

LABELS = ["diabetes", "hypertension", "asthma", "migraine"]
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def b64(path):
    with open(os.path.join(HERE, "figures", path), "rb") as f:
        return base64.b64encode(f.read()).decode()


acc = R["accuracy"]
a_lo, a_hi = R["accuracy_ci95"]
kap, (k_lo, k_hi) = R["kappa"], R["kappa_ci95"]
ac1, (c_lo, c_hi) = R["ac1"], R["ac1_ci95"]
ece = R["ece"]
bl = R["baseline"]
cost = R["cost"]
n = R["n"]

per_class = R["per_class"]
by_len = R["error_by_length"]
by_topic = R["error_by_topic"]
bins = R["calibration_bins"]

payload = json.dumps({
    "labels": LABELS,
    "per_class": per_class,
    "bins": bins,
    "by_len": by_len,
    "by_topic": by_topic,
})

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Can an LLM replace human annotators? | Anurag Nepal</title>
<script src="PLOTLY_CDN"></script>
<style>
  :root { --ink: #1a1a1a; --muted: #555; --accent: #1f77b4; --line: #e2e2e2; }
  body { font-family: Georgia, 'Times New Roman', serif; color: var(--ink);
         max-width: 960px; margin: 0 auto; padding: 24px 20px 60px; line-height: 1.6; }
  h1 { font-size: 1.9rem; margin-bottom: 0.2em; }
  h2 { font-size: 1.35rem; border-bottom: 2px solid var(--ink); padding-bottom: 4px; margin-top: 2.2em; }
  h3 { font-size: 1.1rem; margin-top: 1.6em; }
  .byline { color: var(--muted); font-size: 0.95rem; }
  .statgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 1.2em 0; }
  .stat { border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; background: #fafafa; }
  .stat .v { font-size: 1.5rem; font-weight: bold; color: var(--accent); }
  .stat .l { font-size: 0.85rem; color: var(--muted); }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.95rem; }
  th, td { border: 1px solid var(--line); padding: 7px 10px; text-align: left; }
  th { background: #f2f2f2; }
  img.fig { max-width: 100%; border: 1px solid var(--line); border-radius: 4px; margin: 0.8em 0; }
  .chart { width: 100%; height: 420px; margin: 0.8em 0; }
  .note { background: #fff8e6; border-left: 4px solid #d9a400; padding: 10px 14px; margin: 1em 0; font-size: 0.95rem; }
  footer { margin-top: 3em; color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--line); padding-top: 12px; }
  a { color: var(--accent); }
</style>
</head>
<body>

<h1>Can an LLM replace human annotators?</h1>
<p class="byline">Anurag Nepal &middot; reliability study of Gemini 3.5 Flash-Lite as a
biomedical abstract annotator &middot; October 2026</p>

<h2>1. Overview</h2>
<p>We gave N_ABSTRACTS PubMed abstracts, each tagged with one of four disease
topics by the MeSH query that retrieved it, to Gemini 3.5 Flash-Lite and asked
it to classify each abstract and report a confidence. We then scored the model
the way you would score a new human annotator: accuracy with a confidence
interval, chance-corrected agreement, calibration of its confidence scores, a
comparison against a small supervised baseline, and a cost analysis.</p>

<div class="statgrid">
  <div class="stat"><div class="v">ACC_PCT</div><div class="l">accuracy (95% Wilson CI ACC_LO to ACC_HI)</div></div>
  <div class="stat"><div class="v">KAPPA</div><div class="l">Cohen's kappa (95% bootstrap CI KAPPA_LO to KAPPA_HI)</div></div>
  <div class="stat"><div class="v">AC1V</div><div class="l">Gwet's AC1 (95% bootstrap CI AC1_LO to AC1_HI)</div></div>
  <div class="stat"><div class="v">ECE</div><div class="l">expected calibration error of reported confidences</div></div>
  <div class="stat"><div class="v">COST_LL</div><div class="l">LLM cost per 1,000 annotations (measured)</div></div>
  <div class="stat"><div class="v">COST_HU</div><div class="l">human cost per 1,000 annotations (estimate)</div></div>
</div>

<p><strong>Verdict in one sentence:</strong> the LLM is a strong, very cheap
first-pass annotator for this kind of topic labeling, but its error rate is
too high to replace human adjudication where labels carry real consequences.
The confidence scores can route uncertain items to humans, which is where the
practical value lies.</p>

<h2>2. Methods</h2>
<h3>Data</h3>
<p>N_ABSTRACTS abstracts fetched from PubMed via the Entrez API (October 2026),
using four fixed MeSH queries (diabetes mellitus, hypertension, asthma,
migraine disorders), 62 abstracts per topic. Each abstract's label is the
query that retrieved it: a distant-supervision setup, not a hand-verified
gold standard. Build script: <code>data/build_dataset.py</code> (fixed seed
20261002).</p>
<h3>Annotation protocol</h3>
<p>One API call per abstract, temperature 0, strict JSON output
(<code>{"label", "confidence"}</code>). One-second spacing between calls,
retries with backoff, and every raw response cached to
<code>data/api_cache/</code> before parsing for auditability. PARSE_FAIL
items failed to parse and were excluded from scoring.</p>
<h3>Statistics</h3>
<ul>
  <li><strong>Accuracy</strong> with the Wilson score 95% interval (better than the Wald interval when accuracy is high).</li>
  <li><strong>Cohen's kappa</strong> and <strong>Gwet's AC1</strong>, each with a 2,000-resample percentile bootstrap 95% CI. AC1 is reported alongside kappa because kappa can understate agreement when class marginals are uneven.</li>
  <li><strong>Calibration:</strong> confidences binned into deciles; expected calibration error (ECE) and a calibration curve.</li>
  <li><strong>Baseline:</strong> TF-IDF + logistic regression trained on a labeled 50% split and tested on the same items the LLM labeled, compared with McNemar's exact test on paired predictions. Note the asymmetry: the baseline saw labeled training data; the LLM was zero-shot.</li>
</ul>

<h2>3. Results</h2>

<h3>Overall agreement</h3>
<table>
<tr><th>Metric</th><th>Estimate</th><th>95% CI</th></tr>
<tr><td>Accuracy</td><td>ACC_DEC</td><td>ACC_LO to ACC_HI (Wilson)</td></tr>
<tr><td>Cohen's kappa</td><td>KAPPA</td><td>KAPPA_LO to KAPPA_HI (bootstrap)</td></tr>
<tr><td>Gwet's AC1</td><td>AC1V</td><td>AC1_LO to AC1_HI (bootstrap)</td></tr>
<tr><td>Expected calibration error</td><td>ECE</td><td>--</td></tr>
</table>

<h3>Per-class performance</h3>
<div id="perclass" class="chart"></div>
<img class="fig" src="data:image/png;base64,FIG_CM" alt="Confusion matrix">

<h3>Calibration of confidence scores</h3>
<div id="calib" class="chart"></div>

<h3>Baseline comparison (McNemar)</h3>
<p>On the BASE_N shared test items: baseline TF-IDF + logistic regression
accuracy BASE_ACC, LLM accuracy LLM_TEST_ACC on the same items. Discordant
pairs: baseline-right-only MC_B01, LLM-right-only MC_B10; McNemar exact
p = MC_P. The baseline had the advantage of labeled training data, so the
right reading is not "LLM vs baseline" but "what does a little labeled data
buy you".</p>

<h3>Error analysis</h3>
<div id="errlen" class="chart"></div>
<div id="errtopic" class="chart"></div>

<h3>Cost</h3>
<p>Measured from exact per-call token counts returned by the API
(TOK_IN input tokens, TOK_OUT output tokens) at the published Gemini 3.5
Flash-Lite price of $0.30 / 1M input and $2.50 / 1M output tokens:
<strong>COST_TOTAL</strong> for all N_ABSTRACTS annotations, i.e.
<strong>COST_LL per 1,000</strong>. The human figure (COST_HU per 1,000) is
an estimate based on published per-item rates for simple biomedical labeling
($0.10-$0.50/item; midpoint $0.25 used).</p>
<img class="fig" src="data:image/png;base64,FIG_COST" alt="Cost comparison">

<h2>4. Limitations</h2>
<ul>
  <li><strong>Distant labels, not a gold standard.</strong> The "true" labels are the MeSH queries that retrieved each abstract. Agreement here means agreement with the search query's judgment; a hand-adjudicated set would likely show lower numbers.</li>
  <li><strong>Four well-separated topics.</strong> Diabetes, hypertension, asthma, and migraine have distinctive vocabularies. Harder annotation tasks (fine-grained subtypes, overlapping conditions) would be tougher.</li>
  <li><strong>One model, one prompt, one temperature.</strong> Results describe Gemini 3.5 Flash-Lite with this exact prompt. Prompt changes can move accuracy by several points.</li>
  <li><strong>Publication bias toward clean abstracts.</strong> Records without a substantial abstract were excluded at build time; real annotation pipelines include messy inputs.</li>
  <li><strong>Cost comparison is asymmetric.</strong> The LLM figure is measured; the human figure is an estimate that excludes recruitment, training, and quality-control overhead, which would widen the gap further.</li>
</ul>
<div class="note"><strong>Bottom line:</strong> use the LLM as a cheap first pass with humans reviewing low-confidence or disputed items, not as a drop-in replacement for human annotators.</div>

<footer>
Generated from results.json &middot; code and full write-up at
<a href="https://github.com/nepalanurag/llm-annotator-reliability">github.com/nepalanurag/llm-annotator-reliability</a>
&middot; no external data dependencies on this page.
</footer>

<script id="payload" type="application/json">PAYLOAD</script>
<script>
const D = JSON.parse(document.getElementById("payload").textContent);

// Per-class bars
Plotly.newPlot("perclass", ["precision","recall","f1"].map(m => ({
  x: D.labels, y: D.labels.map(l => D.per_class[l][m]),
  name: m, type: "bar"
})), {barmode: "group", yaxis: {range: [0, 1.05], title: "score"},
      title: "Per-class precision, recall, F1", margin: {t: 40}});

// Calibration curve
const xs = D.bins.map(b => b.mean_confidence), ys = D.bins.map(b => b.accuracy),
      ns = D.bins.map(b => b.n);
Plotly.newPlot("calib", [
  {x: [0,1], y: [0,1], mode: "lines", line: {dash: "dash", color: "gray"}, name: "perfect calibration"},
  {x: xs, y: ys, mode: "markers+text", text: ns.map(v => "n="+v), textposition: "top center",
   marker: {size: ns.map(v => 8 + 24*v/Math.max(...ns)), color: "#1f77b4"}, name: "binned predictions"}
], {xaxis: {range: [0,1], title: "mean predicted confidence"},
    yaxis: {range: [0,1], title: "observed accuracy"},
    title: "Calibration curve", margin: {t: 40}});

// Error by length
Plotly.newPlot("errlen", [{
  x: D.by_len.map(r => r.quartile), y: D.by_len.map(r => r.accuracy),
  error_y: {type: "data",
    array: D.by_len.map(r => r.ci95[1] - r.accuracy),
    arrayminus: D.by_len.map(r => r.accuracy - r.ci95[0])},
  type: "bar", marker: {color: "#1f77b4"}
}], {yaxis: {range: [0, 1.05], title: "accuracy"},
     title: "Accuracy by abstract length quartile (95% Wilson CI)", margin: {t: 40}});

// Error by topic
Plotly.newPlot("errtopic", [{
  x: D.by_topic.map(r => r.topic), y: D.by_topic.map(r => r.accuracy),
  type: "bar", marker: {color: "#ff7f0e"},
  text: D.by_topic.map(r => "n=" + r.n), textposition: "outside"
}], {yaxis: {range: [0, 1.05], title: "accuracy"},
     title: "Accuracy by topic", margin: {t: 40}});
</script>
</body>
</html>
"""

subs = {
    "PLOTLY_CDN": PLOTLY_CDN,
    "N_ABSTRACTS": str(n),
    "ACC_PCT": f"{acc:.1%}",
    "ACC_DEC": f"{acc:.3f}",
    "ACC_LO": f"{a_lo:.3f}", "ACC_HI": f"{a_hi:.3f}",
    "KAPPA": f"{kap:.3f}", "KAPPA_LO": f"{k_lo:.3f}", "KAPPA_HI": f"{k_hi:.3f}",
    "AC1V": f"{ac1:.3f}", "AC1_LO": f"{c_lo:.3f}", "AC1_HI": f"{c_hi:.3f}",
    "ECE": f"{ece:.3f}",
    "COST_LL": f"${cost['llm_cost_per_1000_usd']:.2f}",
    "COST_HU": f"${cost['human_cost_per_1000_usd_estimate']:.0f}",
    "COST_TOTAL": f"${cost['llm_cost_usd_total']:.4f}",
    "TOK_IN": f"{cost['input_tokens_total']:,}",
    "TOK_OUT": f"{cost['output_tokens_total']:,}",
    "PARSE_FAIL": str(R["n_parse_failures"]),
    "BASE_N": str(bl["test_n"]),
    "BASE_ACC": f"{bl['accuracy']:.3f}",
    "LLM_TEST_ACC": f"{bl['llm_accuracy_same_test']:.3f}",
    "MC_B01": str(bl["mcnemar_b01"]), "MC_B10": str(bl["mcnemar_b10"]),
    "MC_P": f"{bl['mcnemar_p']:.4f}",
    "FIG_CM": b64("confusion_matrix.png"),
    "FIG_COST": b64("cost_comparison.png"),
    "PAYLOAD": payload,
}
for k, v in sorted(subs.items(), key=lambda kv: -len(kv[0])):
    html = html.replace(k, v)

out = os.path.join(HERE, "dashboard.html")
with open(out, "w") as f:
    f.write(html)
print("wrote", out, len(html), "bytes")
