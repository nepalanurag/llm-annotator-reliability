"""Label-drift monitoring: compare incoming label distributions to baseline.

Two standard tests, run together:
- PSI (Population Stability Index): sum over labels of
  (actual% - expected%) * ln(actual% / expected%), epsilon-smoothed.
  < 0.10 no significant change; 0.10-0.25 watch; > 0.25 significant shift.
- Chi-square goodness-of-fit of incoming counts vs baseline proportions.

Verdict: ALERT when PSI >= psi_alert_threshold (default 0.25) or the chi-square
test rejects at chi2_alpha. Designed as a scheduled job (see the weekly drift
schedule in pipeline/defs.py); writes a markdown drift report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

from scipy.stats import chisquare

from pipeline import CONTRACT_VERSION
from pipeline.config import load_settings
from pipeline.logging import configure_logging, get_logger

log = get_logger(__name__)

_EPS = 1e-4


def _normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        raise ValueError("distribution must have positive total mass")
    labels = sorted(set(dist))
    return {label: max(dist[label], 0.0) / total for label in labels}


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    """Population Stability Index between two label distributions."""
    e = _normalize(expected)
    a = _normalize(actual)
    labels = sorted(set(e) | set(a))
    total = 0.0
    for label in labels:
        ev = max(e.get(label, 0.0), _EPS)
        av = max(a.get(label, 0.0), _EPS)
        total += (av - ev) * math.log(av / ev)
    return total


def chi2_gof(
    observed_counts: dict[str, int], expected: dict[str, float]
) -> tuple[float, float]:
    """Chi-square goodness-of-fit of observed counts vs expected proportions.

    Returns (statistic, p_value).
    """
    e = _normalize(expected)
    labels = sorted(set(observed_counts) | set(e))
    n = sum(observed_counts.values())
    if n == 0:
        raise ValueError("no observed counts")
    obs = [observed_counts.get(label, 0) for label in labels]
    exp = [max(e.get(label, 0.0), _EPS) * n for label in labels]
    stat, p = chisquare(obs, exp)
    return float(stat), float(p)


def check_drift(
    baseline: dict[str, float],
    incoming_counts: dict[str, int],
    psi_alert: float = 0.25,
    psi_warn: float = 0.10,
    chi2_alpha: float = 0.05,
) -> dict:
    """Run both drift tests; return a verdict dict."""
    n = sum(incoming_counts.values())
    if n == 0:
        raise ValueError("incoming_counts is empty")
    incoming_dist = {k: v / n for k, v in incoming_counts.items()}
    psi_value = psi(baseline, incoming_dist)
    chi2_stat, chi2_p = chi2_gof(incoming_counts, baseline)
    alert = psi_value >= psi_alert or chi2_p < chi2_alpha
    if psi_value >= psi_alert:
        level = "ALERT"
    elif psi_value >= psi_warn or chi2_p < chi2_alpha:
        level = "WATCH"
    else:
        level = "OK"
    verdict = {
        "contract_version": CONTRACT_VERSION,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_incoming": n,
        "psi": round(psi_value, 4),
        "psi_alert_threshold": psi_alert,
        "psi_warning_threshold": psi_warn,
        "chi2_statistic": round(chi2_stat, 3),
        "chi2_p_value": chi2_p,
        "chi2_alpha": chi2_alpha,
        "verdict": level,
        "alert": alert,
        "baseline": _normalize(baseline),
        "incoming": incoming_dist,
    }
    log.info(
        "drift_check", verdict=level, psi=round(psi_value, 4), chi2_p=round(chi2_p, 4)
    )
    return verdict


def write_drift_report(verdict: dict, path: str) -> None:
    """Write a human-readable markdown drift report."""
    lines = [
        "# Label drift report",
        "",
        f"Checked at: {verdict['checked_at']}",
        f"Incoming items: {verdict['n_incoming']}",
        "",
        "## Verdict",
        "",
        f"**{verdict['verdict']}**",
        "",
        "| Test | Value | Threshold |",
        "|---|---|---|",
        f"| PSI | {verdict['psi']} | alert >= {verdict['psi_alert_threshold']}, "
        f"watch >= {verdict['psi_warning_threshold']} |",
        f"| Chi-square p | {verdict['chi2_p_value']:.4f} | "
        f"reject at < {verdict['chi2_alpha']} |",
        "",
        "## Distributions",
        "",
        "| Label | Baseline | Incoming | Delta |",
        "|---|---|---|---|",
    ]
    for label in sorted(set(verdict["baseline"]) | set(verdict["incoming"])):
        b = verdict["baseline"].get(label, 0.0)
        a = verdict["incoming"].get(label, 0.0)
        lines.append(f"| {label} | {b:.3f} | {a:.3f} | {a - b:+.3f} |")
    lines += [
        "",
        "PSI bands: < 0.10 no significant change; 0.10-0.25 watch; "
        "> 0.25 significant shift (standard industry bands).",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("drift_report_written", path=path, verdict=verdict["verdict"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check incoming label counts against a baseline distribution."
    )
    parser.add_argument(
        "--baseline", required=True, help="JSON file: {label: proportion} baseline"
    )
    parser.add_argument(
        "--incoming", required=True, help="JSON file: {label: count} for the new batch"
    )
    parser.add_argument(
        "--report", default=None, help="Write markdown drift report here"
    )
    parser.add_argument("--psi-alert", type=float, default=None)
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit 3 when the verdict is ALERT (for scheduled jobs)",
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)

    for path, kind in ((args.baseline, "baseline"), (args.incoming, "incoming")):
        if not os.path.exists(path):
            print(f"error: {kind} file not found: {path}", file=sys.stderr)
            return 2
    with open(args.baseline, encoding="utf-8") as f:
        baseline_raw = json.load(f)
    # Accept both a bare {label: proportion} map and the versioned baseline
    # file shape {"distribution": {...}, ...}.
    baseline = (
        baseline_raw.get("distribution", baseline_raw)
        if isinstance(baseline_raw, dict)
        else baseline_raw
    )
    with open(args.incoming, encoding="utf-8") as f:
        incoming = json.load(f)
    if not baseline or not incoming:
        print("error: baseline and incoming must both be non-empty", file=sys.stderr)
        return 2

    verdict = check_drift(
        baseline,
        incoming,
        psi_alert=args.psi_alert or settings.psi_alert_threshold,
        psi_warn=settings.psi_warning_threshold,
        chi2_alpha=settings.chi2_alpha,
    )
    if args.report:
        write_drift_report(verdict, args.report)
    print(
        json.dumps(
            {k: v for k, v in verdict.items() if k not in ("baseline", "incoming")},
            indent=2,
        )
    )
    if args.fail_on_alert and verdict["alert"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
