"""Adversarial case generator: deliberately hard annotation cases.

Targets the weakest agreement categories from the repo's findings (REPORT.md
4.5: 9 of 10 errors on the diabetes/hypertension boundary) plus the reviews
axis (sarcastic reviews). Each case carries its intended label, trap type,
and why it is hard.

Modes:
- --dry-run (default): rule-based templates, no API, no cost. Measures the
  RULE backend's brittleness — an honest, labeled proxy, not the LLM's score.
- --llm: Gemini authors the hard cases (needs the custom.google-gemini
  connector credential). This is how you measure the real annotator.

Output: pipeline/adversarial_report.md with per-trap-type agreement breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from pipeline.backends import REVIEW_LABELS, get_backend, load_prompt
from pipeline.config import load_settings
from pipeline.logging import configure_logging, get_logger

log = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "adversarial_report.md"
)

TRAP_TYPES = [
    "boundary_comorbidity",
    "drug_name_trap",
    "symptom_overlap",
    "ambiguous_multi_disease",
    "hedging_language",
    "sarcastic_review",
]

# Templates: (trap_type, intended_label, template, why_hard).
# {disease} slots are filled from the label sets below; the intended label is
# the PRIMARY disease under study, which the surface text obscures.
_TEMPLATES: list[tuple[str, str, str, str]] = [
    (
        "boundary_comorbidity",
        "diabetes",
        "In a cohort of {n} adults with type 2 diabetes and resistant hypertension, "
        "we examined whether intensive blood pressure lowering reduces cardiovascular "
        "events. Glycemic control was maintained with metformin throughout. The primary "
        "outcome was change in HbA1c at 24 months.",
        "Both diseases named; hypertension appears first and dominates the methods, "
        "but the measured outcome is glycemic (diabetes).",
    ),
    (
        "boundary_comorbidity",
        "hypertension",
        "We randomized {n} hypertensive patients with comorbid type 2 diabetes to "
        "intensive vs standard blood pressure targets. Diabetes medications were held "
        "constant. The primary endpoint was systolic blood pressure below 130 mmHg.",
        "Diabetes named as a comorbidity; the endpoint is blood pressure (hypertension).",
    ),
    (
        "drug_name_trap",
        "hypertension",
        "Metformin was co-prescribed in {pct}% of the hypertensive cohort for metabolic "
        "syndrome. After 12 weeks of the antihypertensive regimen, mean systolic pressure "
        "fell by {drop} mmHg. No diabetes diagnosis was present at baseline.",
        "A diabetes drug (metformin) appears in a hypertension study; keyword "
        "annotators latch onto the drug name.",
    ),
    (
        "drug_name_trap",
        "diabetes",
        "Participants with type 2 diabetes received lisinopril for renal protection. "
        "The trial's primary outcome was progression of albuminuria and HbA1c change; "
        "blood pressure was a secondary safety measure only.",
        "An antihypertensive (lisinopril) appears in a diabetes renal-outcome study.",
    ),
    (
        "symptom_overlap",
        "diabetes",
        "Patients reported fatigue, blurred vision, and slow wound healing over {n} "
        "months. Fasting glucose averaged {glu} mg/dL. Fundoscopic examination showed "
        "early diabetic retinopathy.",
        "Fatigue and blurred vision overlap with hypertension; the discriminating "
        "evidence (glucose, retinopathy) sits late in the abstract.",
    ),
    (
        "symptom_overlap",
        "hypertension",
        "The patient presented with headache, dizziness, and fatigue. Ambulatory "
        "monitoring showed mean 24-hour pressure of {bp} mmHg with a non-dipping "
        "pattern. No hyperglycemia was documented.",
        "Headache/dizziness/fatigue are shared symptoms; only the pressure reading "
        "discriminates.",
    ),
    (
        "ambiguous_multi_disease",
        "asthma",
        "In patients with chronic kidney disease, we compared inhaled corticosteroid "
        "response in those with and without comorbid diabetes. Asthma control "
        "questionnaire scores were the primary outcome in the {n}-patient substudy.",
        "Three diseases named (kidney disease, diabetes, asthma); the outcome is "
        "asthma control.",
    ),
    (
        "ambiguous_multi_disease",
        "migraine",
        "Among {n} emergency visits for headache with hypertensive urgency, we tested "
        "whether migraine history predicts admission. Triptan use was documented in "
        "{pct}% of cases with a final migraine diagnosis.",
        "Hypertension and headache co-occur; the classification target is migraine.",
    ),
    (
        "hedging_language",
        "diabetes",
        "These findings may suggest a possible association between the observed "
        "metabolic pattern and diabetes-related pathways, although the evidence "
        "remains inconclusive and further study is warranted.",
        "Hedged language with no definitive disease statement; annotators must "
        "commit despite the uncertainty.",
    ),
    (
        "hedging_language",
        "hypertension",
        "Blood pressure trends were suggestive but not conclusive of a treatment "
        "effect; the authors note the association with hypertension outcomes "
        "requires confirmation in larger samples.",
        "Same hedging pattern on the hypertension side.",
    ),
    (
        "sarcastic_review",
        "negative",
        "Oh great, another phone charger that works for exactly two days before "
        "dying. I just LOVE buying the same thing three times. Five stars for "
        "consistency, I guess.",
        "Sarcastic positive surface sentiment with negative intent; keyword "
        "annotators count 'love' and 'five stars'.",
    ),
    (
        "sarcastic_review",
        "negative",
        "Fantastic product. It broke the first week, customer service was super "
        "helpful (they hung up on me twice), and I would absolutely recommend it "
        "to my worst enemy.",
        "Praise words used ironically throughout; true label is negative.",
    ),
    (
        "sarcastic_review",
        "positive",
        "Yeah, the box was ugly and shipping took forever, but honestly this is "
        "the best purchase I have made all year. Works perfectly, zero complaints "
        "about the actual product.",
        "Negative surface details with a genuinely positive verdict; tests whether "
        "the annotator weighs the conclusion over the complaints.",
    ),
]

_SENTINEL_LABELS = {
    "boundary_comorbidity",
    "drug_name_trap",
    "symptom_overlap",
    "ambiguous_multi_disease",
    "hedging_language",
}
_REVIEW_TRAPS = {"sarcastic_review"}


def generate_dry_run(seed: int = 20261008, per_trap: int = 4) -> list[dict]:
    """Generate hard cases from templates. Deterministic given seed."""
    rng = random.Random(seed)
    cases: list[dict] = []
    idx = 0
    for trap, intended, template, why in _TEMPLATES:
        for rep in range(per_trap):
            text = template.format(
                n=rng.choice([120, 240, 480, 960]),
                pct=rng.choice([18, 34, 47, 62]),
                drop=rng.choice([8, 11, 14]),
                glu=rng.choice([168, 192, 214]),
                bp=rng.choice(["152/96", "148/94", "161/99"]),
            )
            # Vary surface wording slightly so near-dup detection stays honest.
            cases.append(
                {
                    "case_id": f"adv-{idx:03d}",
                    "trap_type": trap,
                    "intended_label": intended,
                    "label_set": ("reviews" if trap in _REVIEW_TRAPS else "diseases"),
                    "text": text,
                    "why_hard": why,
                    "generator": "dry-run-templates",
                    "seed": seed,
                }
            )
            idx += 1
    rng.shuffle(cases)
    return cases


def generate_llm(n_per_trap: int = 4, model_note: str = "gemini") -> list[dict]:
    """Ask Gemini to author hard cases. Needs the connector credential."""
    from pipeline.backends import GeminiBackend

    backend = GeminiBackend()
    template, _ = load_prompt("v1")
    cases: list[dict] = []
    idx = 0
    for trap in TRAP_TYPES:
        label_set = (
            REVIEW_LABELS
            if trap in _REVIEW_TRAPS
            else ["diabetes", "hypertension", "asthma", "migraine"]
        )
        prompt = (
            "You are writing adversarial test cases for a text classifier. "
            f"Write {n_per_trap} short abstract-like texts (2-4 sentences each) that are "
            f"deliberately hard to classify because of this trap: {trap}. "
            f"Labels must come from {label_set}. "
            "Reply with ONLY a JSON array of objects with keys "
            "'text', 'intended_label', 'why_hard'."
        )
        raw, _ = backend._call(prompt, cache_key=f"adversarial-llm:{trap}:{n_per_trap}")
        try:
            parts = raw["candidates"][0]["content"]["parts"]
            text_out = next(p["text"] for p in parts if "text" in p)
            items = json.loads(text_out)
        except Exception as exc:
            raise RuntimeError(
                f"LLM case generation failed for trap {trap}: {exc}"
            ) from exc
        for item in items:
            cases.append(
                {
                    "case_id": f"adv-llm-{idx:03d}",
                    "trap_type": trap,
                    "intended_label": str(item["intended_label"]).strip().lower(),
                    "label_set": "reviews" if trap in _REVIEW_TRAPS else "diseases",
                    "text": item["text"],
                    "why_hard": item.get("why_hard", ""),
                    "generator": f"llm-{model_note}",
                }
            )
            idx += 1
    return cases


def evaluate(cases: list[dict], backend_name: str, prompt_version: str) -> dict:
    """Run the annotator over adversarial cases; per-trap-type breakdown."""
    backend = get_backend(backend_name)
    disease_labels = ["diabetes", "hypertension", "asthma", "migraine"]
    by_trap: dict[str, dict] = {}
    for case in cases:
        labels = REVIEW_LABELS if case["label_set"] == "reviews" else disease_labels
        (ann,) = backend.annotate(
            [(case["case_id"], case["text"])], labels, prompt_version
        )
        correct = ann.label == case["intended_label"]
        slot = by_trap.setdefault(
            case["trap_type"], {"n": 0, "correct": 0, "misses": []}
        )
        slot["n"] += 1
        slot["correct"] += int(correct)
        if not correct:
            slot["misses"].append(
                {
                    "case_id": case["case_id"],
                    "intended": case["intended_label"],
                    "predicted": ann.label,
                    "confidence": ann.confidence,
                    "why_hard": case["why_hard"],
                }
            )
    for trap, slot in by_trap.items():
        slot["accuracy"] = round(slot["correct"] / slot["n"], 3)
    return by_trap


def write_report(
    by_trap: dict, cases: list[dict], backend_name: str, mode: str, path: str
) -> None:
    total_n = sum(s["n"] for s in by_trap.values())
    total_c = sum(s["correct"] for s in by_trap.values())
    weakest = min(by_trap.items(), key=lambda kv: kv[1]["accuracy"])
    lines = [
        "# Adversarial annotation report",
        "",
        f"Mode: {mode} | Backend under test: {backend_name} | Cases: {total_n}",
        "",
        "## What this measures",
        "",
        "Hard cases target the weakest agreement categories from the repo's "
        "error analysis (REPORT.md 4.5): the diabetes/hypertension boundary, "
        "drug-name traps, and symptom overlap — plus sarcastic reviews for the "
        "reviews axis. In --dry-run mode the cases are rule-generated templates "
        "and the backend under test is the keyword baseline, so this report "
        "measures the *baseline's* brittleness, not the LLM's. That is an honest "
        "proxy: it tells you which trap types a naive annotator falls for. "
        "To measure the real annotator, run with --llm (needs the "
        "custom.google-gemini connector credential).",
        "",
        "## Per-trap-type agreement",
        "",
        "| Trap type | n | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for trap in TRAP_TYPES:
        s = by_trap.get(trap)
        if s:
            lines.append(
                f"| {trap} | {s['n']} | {s['correct']} | {s['accuracy']:.3f} |"
            )
    lines += [
        "",
        f"Overall: {total_c}/{total_n} = {total_c / total_n:.3f}",
        "",
        f"## Weakest trap: {weakest[0]} (accuracy {weakest[1]['accuracy']:.3f})",
        "",
    ]
    for m in weakest[1]["misses"][:6]:
        lines += [
            f"- {m['case_id']}: intended **{m['intended']}**, predicted "
            f"**{m['predicted']}** (conf {m['confidence']}). {m['why_hard']}",
        ]
    lines += [
        "",
        "## Reading the misses",
        "",
    ]
    broken = [(t, s) for t, s in by_trap.items() if s["accuracy"] < 0.75]
    held = [(t, s) for t, s in by_trap.items() if s["accuracy"] >= 0.75]
    if broken:
        lines.append(
            "Trap types that broke this backend "
            + "(accuracy < 0.75): "
            + ", ".join(
                f"{t} ({s['accuracy']:.2f})"
                for t, s in sorted(broken, key=lambda kv: kv[1]["accuracy"])
            )
            + ". These are the blind spots to fix first: for sarcastic reviews "
            "the backend needs pragmatic (not lexical) cues; for symptom "
            "overlap and multi-disease ambiguity it needs the v2 prompt's "
            "tie-break rule ('primary disease under study') or a stronger model."
        )
    if held:
        lines.append(
            "Trap types this backend survived "
            + "(accuracy >= 0.75): "
            + ", ".join(f"{t} ({s['accuracy']:.2f})" for t, s in held)
            + ". Note this is backend-specific: the boundary and drug-name "
            "traps are designed around the failure mode the repo's error "
            "analysis found in the *LLM* (9 of 10 errors on the "
            "diabetes/hypertension boundary). The keyword backend survives them "
            "here only because the templates mention the true disease's "
            "vocabulary more often. Run with --llm to measure the real "
            "annotator on these traps."
        )
    lines += [
        "",
        "## Case inventory",
        "",
        "| case_id | trap_type | intended | generator |",
        "|---|---|---|---|",
    ]
    for c in cases:
        lines.append(
            f"| {c['case_id']} | {c['trap_type']} | {c['intended_label']} "
            f"| {c['generator']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(
        "adversarial_report_written",
        path=path,
        overall_accuracy=round(total_c / total_n, 3),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate adversarial annotation cases and test a backend on them."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Rule-based cases, no API (default)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Gemini authors the cases (needs connector credential)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["rule", "gemini"],
        help="Backend under test (default: config)",
    )
    parser.add_argument("--prompt", default=None, dest="prompt_version")
    parser.add_argument("--per-trap", type=int, default=4, help="Cases per trap type")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)

    if args.per_trap < 1:
        print("error: --per-trap must be >= 1", file=sys.stderr)
        return 2
    backend_name = args.backend or settings.backend
    prompt_version = args.prompt_version or settings.prompt_version

    if args.llm:
        mode = "llm"
        cases = generate_llm(n_per_trap=args.per_trap)
    else:
        mode = "dry-run"
        cases = generate_dry_run(seed=args.seed, per_trap=args.per_trap)
    log.info("adversarial_cases_generated", mode=mode, n=len(cases))
    by_trap = evaluate(cases, backend_name, prompt_version)
    write_report(by_trap, cases, backend_name, mode, args.report)
    print(f"adversarial report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
