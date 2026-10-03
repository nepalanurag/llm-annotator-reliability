"""Agreement and reliability statistics for the LLM-annotator study.

All functions are pure numpy. CIs use the Wilson interval (accuracy) or the
percentile bootstrap (kappa, AC1). Fixed seeds make every number reproducible.
"""
import numpy as np

SEED = 20261002


def wilson_ci(n_success, n_total, alpha=0.05):
    """Wilson score interval for a binomial proportion."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = n_success / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denom
    return float(center - half), float(center + half)


def cohens_kappa(y_true, y_pred, labels):
    """Cohen's kappa for multi-class agreement (unweighted)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    label_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[label_idx[t], label_idx[p]] += 1
    po = np.trace(cm) / n
    pe = float(np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / n ** 2)
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def gwet_ac1(y_true, y_pred, labels):
    """Gwet's AC1 for multi-class agreement.

    For K classes, chance agreement is computed per category as
    pe = mean over categories of pi*(1-pi)/(1-1/K) where pi is the average of
    the two raters' marginal proportions, then kappa = (po - pe)/(1 - pe).
    (Gwet 2008, "Computing inter-rater reliability and its variance in the
    presence of high agreement".)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    K = len(labels)
    label_idx = {lab: i for i, lab in enumerate(labels)}
    p_true = np.array([np.mean(y_true == lab) for lab in labels])
    p_pred = np.array([np.mean(y_pred == lab) for lab in labels])
    po = float(np.mean(y_true == y_pred))
    pi = (p_true + p_pred) / 2.0
    if K <= 1:
        return 1.0
    pe = float(np.sum(pi * (1 - pi)) / (K - 1))
    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def bootstrap_ci(stat_fn, y_true, y_pred, labels, n_boot=2000, seed=SEED):
    """Percentile bootstrap 95% CI for a statistic of (y_true, y_pred)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat_fn(y_true[idx], y_pred[idx], labels)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(b01, b10):
    """McNemar's exact test for paired binary outcomes.

    b01 = cases where method 1 wrong and method 2 right,
    b10 = cases where method 1 right and method 2 wrong.
    Returns (odds_ratio, p_value).
    """
    from scipy.stats import binomtest
    n = b01 + b10
    res = binomtest(b01, n, 0.5)
    oratio = (b01 / b10) if b10 else float("inf")
    return float(oratio), float(res.pvalue)


def calibration_bins(confidences, correct, n_bins=10):
    """Bin confidences; return per-bin (mean_conf, accuracy, count)."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i else \
               (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bins.append({
            "bin": i,
            "lo": float(lo), "hi": float(hi),
            "n": int(mask.sum()),
            "mean_confidence": float(confidences[mask].mean()),
            "accuracy": float(correct[mask].mean()),
        })
    return bins


def expected_calibration_error(confidences, correct, n_bins=10):
    """ECE = sum over bins |accuracy - mean_confidence| * bin_weight."""
    bins = calibration_bins(confidences, correct, n_bins)
    n = len(confidences)
    return float(sum(b["n"] / n * abs(b["accuracy"] - b["mean_confidence"])
                     for b in bins))


def per_class_metrics(y_true, y_pred, labels):
    """Precision, recall, F1 per class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for lab in labels:
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1,
                    "support": int(np.sum(y_true == lab))}
    return out
