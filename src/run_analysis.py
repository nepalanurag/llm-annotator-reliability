"""Full statistical analysis for the LLM-annotator reliability study.

Loads data/abstracts.csv + data/annotations.csv, computes every reported
statistic, writes figures/ PNGs and results.json (the single source of truth
for the numbers that go into the README, REPORT, notebook, and dashboard).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from metrics import (  # noqa: E402
    wilson_ci, cohens_kappa, gwet_ac1, bootstrap_ci, mcnemar,
    calibration_bins, expected_calibration_error, per_class_metrics)

SEED = 20261002
LABELS = ["diabetes", "hypertension", "asthma", "migraine"]

# Gemini 3.5 Flash-Lite list pricing (Google, verified Oct 2026):
# $0.30 / 1M input tokens, $2.50 / 1M output tokens.
PRICE_IN_1M = 0.30
PRICE_OUT_1M = 2.50


def main():
    rng = np.random.default_rng(SEED)
    abstracts = pd.read_csv(os.path.join(HERE, "data", "abstracts.csv"))
    ann = pd.read_csv(os.path.join(HERE, "data", "annotations.csv"))
    df = abstracts.merge(ann, on=["pmid", "topic"], how="inner")
    df = df[df["parsed_ok"] == 1].copy().reset_index(drop=True)

    y_true = df["topic"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    n = len(df)

    acc = float(np.mean(y_true == y_pred))
    acc_lo, acc_hi = wilson_ci(int((y_true == y_pred).sum()), n)
    kappa = cohens_kappa(y_true, y_pred, LABELS)
    kappa_lo, kappa_hi = bootstrap_ci(cohens_kappa, y_true, y_pred, LABELS)
    ac1 = gwet_ac1(y_true, y_pred, LABELS)
    ac1_lo, ac1_hi = bootstrap_ci(gwet_ac1, y_true, y_pred, LABELS)
    per_class = per_class_metrics(y_true, y_pred, LABELS)

    conf = df["confidence"].to_numpy()
    correct = (y_true == y_pred).astype(float)
    ece = expected_calibration_error(conf, correct, n_bins=10)
    cal_bins = calibration_bins(conf, correct, n_bins=10)

    # --- TF-IDF + logistic regression baseline, same test items the LLM saw.
    # Train on a 75/25 split; the baseline learns from labeled training data
    # while the LLM is zero-shot, so this is zero-shot LLM vs a small
    # supervised model, not an apples-to-apples benchmark.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.5, random_state=SEED,
                              stratify=y_true)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = vec.fit_transform(df.loc[tr, "text"])
    Xte = vec.transform(df.loc[te, "text"])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(Xtr, y_true[tr])
    base_pred = clf.predict(Xte)
    base_acc = float(np.mean(base_pred == y_true[te]))
    base_lo, base_hi = wilson_ci(int((base_pred == y_true[te]).sum()), len(te))
    llm_te = y_pred[te]
    # McNemar on the same test items: discordant pairs
    llm_ok = (llm_te == y_true[te])
    base_ok = (base_pred == y_true[te])
    b01 = int(np.sum(~llm_ok & base_ok))   # baseline right, LLM wrong
    b10 = int(np.sum(llm_ok & ~base_ok))   # LLM right, baseline wrong
    mc_or, mc_p = mcnemar(b01, b10)

    # --- Error analysis by text length quartile and by topic
    df["correct"] = (y_true == y_pred).astype(int)
    df["len_quartile"] = pd.qcut(df["n_words"], 4,
                                 labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])
    by_len = df.groupby("len_quartile", observed=True)["correct"].agg(
        ["mean", "count"]).reset_index()
    by_len_ci = [wilson_ci(int((df[df["len_quartile"] == q]["correct"]).sum()),
                           int((df["len_quartile"] == q).sum()))
                 for q in by_len["len_quartile"]]
    by_topic = df.groupby("topic")["correct"].agg(["mean", "count"]).reset_index()

    # --- Cost
    tot_in = int(df["prompt_tokens"].sum())
    tot_out = int(df["output_tokens"].sum())
    llm_cost_total = tot_in / 1e6 * PRICE_IN_1M + tot_out / 1e6 * PRICE_OUT_1M
    llm_cost_per_1k = llm_cost_total / n * 1000
    # Human annotation estimate: expert biomedical annotation, widely cited
    # around $0.10-$0.50 per abstract for simple classification tasks;
    # we use $0.25/item as a mid-point labeled estimate.
    human_cost_per_item = 0.25
    human_cost_per_1k = human_cost_per_item * 1000

    results = {
        "n": n,
        "n_parse_failures": int((ann["parsed_ok"] == 0).sum()),
        "seed": SEED,
        "model": "models/gemini-3.5-flash-lite",
        "accuracy": acc,
        "accuracy_ci95": [acc_lo, acc_hi],
        "kappa": kappa,
        "kappa_ci95": [kappa_lo, kappa_hi],
        "ac1": ac1,
        "ac1_ci95": [ac1_lo, ac1_hi],
        "per_class": per_class,
        "ece": ece,
        "calibration_bins": cal_bins,
        "baseline": {
            "name": "TF-IDF + logistic regression (trained on 50% labeled split)",
            "test_n": int(len(te)),
            "accuracy": base_acc,
            "accuracy_ci95": [base_lo, base_hi],
            "llm_accuracy_same_test": float(np.mean(llm_te == y_true[te])),
            "mcnemar_b01": b01, "mcnemar_b10": b10,
            "mcnemar_odds_ratio": mc_or, "mcnemar_p": mc_p,
        },
        "error_by_length": [
            {"quartile": str(r["len_quartile"]), "accuracy": float(r["mean"]),
             "n": int(r["count"]), "ci95": [float(c[0]), float(c[1])]}
            for r, c in zip(by_len.to_dict("records"), by_len_ci)],
        "error_by_topic": [
            {"topic": r["topic"], "accuracy": float(r["mean"]), "n": int(r["count"])}
            for r in by_topic.to_dict("records")],
        "cost": {
            "input_tokens_total": tot_in,
            "output_tokens_total": tot_out,
            "llm_cost_usd_total": llm_cost_total,
            "llm_cost_per_1000_usd": llm_cost_per_1k,
            "price_input_per_1m": PRICE_IN_1M,
            "price_output_per_1m": PRICE_OUT_1M,
            "human_cost_per_item_usd_estimate": human_cost_per_item,
            "human_cost_per_1000_usd_estimate": human_cost_per_1k,
        },
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Figures
    sys.path.insert(0, os.path.join(HERE, "src"))
    from plots import (confusion_matrix_fig, calibration_fig,  # noqa: E402
                       per_class_fig, cost_fig)
    confusion_matrix_fig(y_true, y_pred, LABELS,
                         os.path.join(HERE, "figures", "confusion_matrix.png"))
    calibration_fig(cal_bins, ece,
                    os.path.join(HERE, "figures", "calibration.png"))
    per_class_fig(per_class, LABELS,
                  os.path.join(HERE, "figures", "per_class_metrics.png"))
    cost_fig(llm_cost_per_1k, human_cost_per_1k,
             os.path.join(HERE, "figures", "cost_comparison.png"))

    print(json.dumps({
        "n": n, "accuracy": acc, "acc_ci": [acc_lo, acc_hi],
        "kappa": kappa, "kappa_ci": [kappa_lo, kappa_hi],
        "ac1": ac1, "ac1_ci": [ac1_lo, ac1_hi],
        "ece": ece, "mcnemar_p": mc_p,
        "baseline_acc": base_acc, "llm_same_test": float(np.mean(llm_te == y_true[te])),
        "llm_cost_per_1k": llm_cost_per_1k,
    }, indent=2))


if __name__ == "__main__":
    main()
