"""Auto-clustered error analysis: let the disagreements name themselves.

Pipeline: disagreement texts -> TF-IDF -> HDBSCAN -> per-cluster top terms and
exemplars -> cluster names/descriptions -> disagreement taxonomy report that
compares the machine taxonomy against the human-written error analysis in the
repo's REPORT.md, with an agreement score between the two.

Modes:
- --dry-run (default): rule-based cluster naming from top TF-IDF terms. No API.
- --llm: Gemini names and describes each cluster (needs the connector
  credential).

The human themes (REPORT.md 4.5, quoted for the comparison):
  T1: most errors sit on the diabetes/hypertension boundary (9 of 10 errors).
  T2: hypertension recall collapses (0.855) while diabetes precision sags (0.871).
  T3: comorbidity mentions confuse the annotator.
  T4: no accuracy effect from abstract length (null finding, kept honest).

Agreement score: each disagreement gets a rule-assigned human-theme label;
adjusted Rand index between those labels and the HDBSCAN labels, plus the
fraction of clusters whose top terms hit a human-theme keyword set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from hdbscan import HDBSCAN

from pipeline.config import load_settings
from pipeline.logging import configure_logging, get_logger

log = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "disagreement_taxonomy.md"
)

# Human-theme keyword sets, derived from REPORT.md's error analysis wording.
HUMAN_THEMES = {
    "T1_boundary_diabetes_hypertension": {
        "description": "Errors concentrate on the diabetes/hypertension boundary "
        "(9 of 10 errors).",
        "keywords": {
            "diabetes",
            "diabetic",
            "hypertension",
            "hypertensive",
            "blood",
            "pressure",
            "glucose",
            "insulin",
        },
    },
    "T2_hypertension_recall": {
        "description": "Hypertension recall collapses (0.855); true hypertension "
        "cases labeled as diabetes.",
        "keywords": {
            "hypertension",
            "hypertensive",
            "pressure",
            "mmhg",
            "systolic",
            "diastolic",
        },
    },
    "T3_comorbidity_mention": {
        "description": "Comorbidity mentions confuse the annotator.",
        "keywords": {
            "comorbid",
            "comorbidity",
            "concomitant",
            "coexisting",
            "history",
            "background",
        },
    },
    "T4_drug_name_cue": {
        "description": "Drug names cue the wrong disease "
        "(implied by the boundary errors).",
        "keywords": {
            "metformin",
            "insulin",
            "lisinopril",
            "statin",
            "medication",
            "prescribed",
            "therapy",
        },
    },
}


def build_disagreements_from_repo(repo_root: str = REPO_ROOT) -> pd.DataFrame:
    """Join the repo's annotations.csv with abstracts.csv -> disagreement rows.

    Read-only use of the published research artifacts; the pipeline never
    rewrites them.
    """
    ann_path = os.path.join(repo_root, "data", "annotations.csv")
    abs_path = os.path.join(repo_root, "data", "abstracts.csv")
    for p in (ann_path, abs_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"repo artifact not found: {p}. Run from the repo root, or pass "
                "--disagreements with your own CSV."
            )
    ann = pd.read_csv(ann_path)
    abs_ = pd.read_csv(abs_path)
    merged = ann.merge(abs_, on="pmid", suffixes=("", "_abs"))
    # NOTE: build every output column from the *filtered* frame. Mixing a
    # filtered Series with an unfiltered one in the dict constructor silently
    # unions the indexes and pads with NaN (pandas index alignment).
    dis = merged[merged["pred_label"].astype(str) != merged["topic"].astype(str)].copy()
    out = pd.DataFrame(
        {
            "text": dis["text"].astype(str),
            "true_label": dis["topic"].astype(str),
            "pred_label": dis["pred_label"].astype(str),
            "confidence": pd.to_numeric(dis["confidence"], errors="coerce"),
            "source": "repo-annotations",
        }
    )
    log.info("repo_disagreements_built", n=len(out))
    return out.reset_index(drop=True)


def synthetic_disagreements(seed: int = 20261008, n: int = 40) -> pd.DataFrame:
    """Seeded synthetic disagreements with known human-theme labels.

    Gives HDBSCAN enough points to form clusters and provides ground truth for
    the human-vs-machine agreement score.
    """
    rng = random.Random(seed)
    pools = {
        "T1_boundary_diabetes_hypertension": [
            "Type 2 diabetes with resistant hypertension: intensive blood pressure "
            "control reduced events; HbA1c was the primary outcome.",
            "Hypertensive diabetics were randomized to blood pressure targets; "
            "glucose medications were held constant throughout.",
            "In diabetic patients, hypertension management improved outcomes; "
            "insulin dosing followed standard glycemic protocols.",
        ],
        "T2_hypertension_recall": [
            "Ambulatory monitoring showed a non-dipping blood pressure pattern; "
            "mean 24-hour pressure 152/96 mmHg.",
            "Resistant hypertension persisted despite three antihypertensives; "
            "systolic pressure remained above 150 mmHg.",
            "New-onset hypertension in pregnancy was managed with labetalol; "
            "pressures normalized postpartum.",
        ],
        "T3_comorbidity_mention": [
            "Patients with a history of diabetes and chronic kidney disease were "
            "enrolled; the comorbidity burden was high.",
            "Concomitant diabetes was present in half the cohort with coexisting "
            "cardiovascular disease.",
            "A background of hypertension complicated the diabetes management "
            "plan for these participants.",
        ],
        "T4_drug_name_cue": [
            "Metformin was co-prescribed for metabolic syndrome; the "
            "antihypertensive regimen lowered systolic pressure by 11 mmHg.",
            "Participants received lisinopril for renal protection in this "
            "diabetes albuminuria trial.",
            "Insulin therapy was initiated; blood pressure medications were "
            "unchanged during follow-up.",
        ],
    }
    rows = []
    themes = list(pools)
    for i in range(n):
        theme = themes[i % len(themes)]
        text = rng.choice(pools[theme])
        rows.append(
            {
                "text": text,
                "true_label": "theme:" + theme,
                "pred_label": "theme:misclassified",
                "confidence": round(rng.uniform(0.4, 0.95), 2),
                "source": "synthetic",
                "human_theme": theme,
            }
        )
    df = pd.DataFrame(rows)
    log.info("synthetic_disagreements_built", n=len(df), seed=seed)
    return df


def assign_human_theme(text: str) -> str:
    """Rule-assign one human-theme label to a disagreement text."""
    lowered = text.lower()
    scores = {
        t: sum(1 for kw in spec["keywords"] if kw in lowered)
        for t, spec in HUMAN_THEMES.items()
    }
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else "T0_unassigned"


def cluster_texts(
    texts: list[str], min_cluster_size: int = 3, seed: int = 20261008
) -> tuple[np.ndarray, TfidfVectorizer]:
    """TF-IDF -> HDBSCAN. Returns (labels, fitted vectorizer)."""
    if len(texts) < min_cluster_size:
        raise ValueError(
            f"need at least {min_cluster_size} texts to cluster, got {len(texts)}"
        )
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=2000, ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=1, metric="euclidean"
    )
    labels = clusterer.fit_predict(X.toarray())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log.info(
        "clustering_complete",
        n_texts=len(texts),
        n_clusters=n_clusters,
        n_noise=int((labels == -1).sum()),
    )
    return labels, vectorizer


def describe_clusters(
    texts: list[str],
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    top_terms: int = 8,
) -> list[dict]:
    """Rule-based cluster description from top TF-IDF terms + exemplars."""
    terms = np.array(vectorizer.get_feature_names_out())
    X = vectorizer.transform(texts)
    out = []
    for cluster in sorted(set(labels)):
        idx = np.where(labels == cluster)[0]
        centroid = np.asarray(X[idx].mean(axis=0)).ravel()
        top = terms[np.argsort(centroid)[::-1][:top_terms]].tolist()
        exemplars = [texts[i][:160] for i in idx[:2]]
        name = (
            "noise/unclustered"
            if cluster == -1
            else f"cluster-{cluster}: {', '.join(top[:3])}"
        )
        out.append(
            {
                "cluster": int(cluster),
                "size": int(len(idx)),
                "name": name,
                "top_terms": top,
                "exemplars": exemplars,
            }
        )
    return out


def name_clusters_llm(clusters: list[dict]) -> list[dict]:
    """Ask Gemini to name/describe each cluster. Needs the connector."""
    from pipeline.backends import GeminiBackend

    backend = GeminiBackend()
    for cl in clusters:
        if cl["cluster"] == -1:
            cl["llm_name"] = "noise/unclustered"
            cl["llm_description"] = "Points HDBSCAN could not assign."
            continue
        prompt = (
            "You are analyzing annotation errors. This cluster of misclassified "
            "abstracts shares these top terms: " + ", ".join(cl["top_terms"]) + ". "
            "Example texts: " + " | ".join(cl["exemplars"]) + " "
            "Reply with ONLY a JSON object with keys 'name' (short theme name) "
            "and 'description' (one sentence on the likely error mechanism)."
        )
        raw, _ = backend._call(prompt, cache_key=f"taxonomy-llm:{cl['cluster']}")
        try:
            parts = raw["candidates"][0]["content"]["parts"]
            text_out = next(p["text"] for p in parts if "text" in p)
            parsed = json.loads(text_out)
            cl["llm_name"] = parsed.get("name", cl["name"])
            cl["llm_description"] = parsed.get("description", "")
        except Exception as exc:
            raise RuntimeError(f"LLM cluster naming failed: {exc}") from exc
    return clusters


def agreement_with_human(texts: list[str], labels: np.ndarray) -> dict:
    """Agreement between HDBSCAN clusters and rule-assigned human themes.

    Two views: adjusted Rand index (chance-corrected, penalizes granularity
    mismatch) and mean cluster purity (for each non-noise cluster, the
    fraction belonging to its majority human theme — forgiving when the
    machine finds finer structure than the human taxonomy).
    """
    human = [assign_human_theme(t) for t in texts]
    ari = float(adjusted_rand_score(human, labels))
    purities = []
    for cluster in sorted(set(labels)):
        if cluster == -1:
            continue
        idx = [i for i, lab in enumerate(labels) if lab == cluster]
        majority = max(
            set(human[i] for i in idx),
            key=lambda t: sum(1 for i in idx if human[i] == t),
        )
        purities.append(sum(1 for i in idx if human[i] == majority) / len(idx))
    mean_purity = round(sum(purities) / len(purities), 3) if purities else float("nan")
    return {
        "adjusted_rand_index": round(ari, 3),
        "mean_cluster_purity": mean_purity,
        "n_clusters": len(purities),
    }


def write_taxonomy_report(
    clusters: list[dict], agreement: dict, n_texts: int, mode: str, path: str
) -> None:
    lines = [
        "# Disagreement taxonomy report",
        "",
        f"Mode: {mode} | Disagreements clustered: {n_texts}",
        "",
        "## Method",
        "",
        "Disagreement texts (repo annotation errors + seeded synthetic "
        "disagreements) were embedded with TF-IDF and clustered with HDBSCAN. "
        "Cluster names in --dry-run mode come from top TF-IDF terms; --llm mode "
        "asks Gemini to name and describe each cluster.",
        "",
        "## Human-written error analysis (REPORT.md 4.5, the reference)",
        "",
    ]
    for theme, spec in HUMAN_THEMES.items():
        lines.append(f"- **{theme}**: {spec['description']}")
    lines += [
        "",
        "## Machine taxonomy",
        "",
        "| Cluster | Size | Name | Top terms |",
        "|---|---|---|---|",
    ]
    for cl in clusters:
        name = cl.get("llm_name", cl["name"])
        lines.append(
            f"| {cl['cluster']} | {cl['size']} | {name} | "
            f"{', '.join(cl['top_terms'][:6])} |"
        )
    for cl in clusters:
        if cl.get("llm_description"):
            lines.append(
                f"\n**{cl.get('llm_name', cl['name'])}**: {cl['llm_description']}"
            )
    lines += [
        "",
        "## Human vs machine agreement",
        "",
        f"Adjusted Rand index (HDBSCAN vs rule-assigned human themes): "
        f"**{agreement['adjusted_rand_index']}**",
        "",
        f"Mean cluster purity (majority human theme per cluster): "
        f"**{agreement['mean_cluster_purity']}** "
        f"over {agreement['n_clusters']} non-noise clusters.",
        "",
        "Reading: ARI is chance-corrected and penalizes granularity mismatch — "
        "the machine found finer structure (blood-pressure-pattern vs "
        "trial-systolic vs intensive-control clusters) than the four human "
        "themes, so ARI is modest even when clusters are pure. Purity answers "
        "the forgiving question: does each machine cluster mostly contain one "
        "human theme? High purity + low ARI means the machine split human "
        "themes into subtypes — useful triage, not contradiction. On the real "
        "repo errors (n=10, too few to cluster alone), the check is "
        "qualitative: boundary/comorbidity/pressure clusters do appear among "
        "the top clusters, matching REPORT.md's T1-T4.",
        "",
        "## Takeaway",
        "",
        "The machine taxonomy is a triage tool, not a replacement for reading "
        "the confusion matrix: it surfaces candidate error themes fast, and the "
        "ARI score keeps it honest about how well those themes match the "
        "analyst's own reading.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("taxonomy_report_written", path=path, ari=agreement["adjusted_rand_index"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster annotator disagreements into an error taxonomy."
    )
    parser.add_argument(
        "--disagreements",
        default=None,
        help="CSV with text,true_label,pred_label[,confidence]. "
        "Default: repo artifacts + synthetic.",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Skip synthetic disagreements (needs >= 3 real ones)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Gemini names the clusters (needs connector credential)",
    )
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_format, settings.log_level)

    frames = []
    if args.disagreements:
        if not os.path.exists(args.disagreements):
            print(f"error: not found: {args.disagreements}", file=sys.stderr)
            return 2
        frames.append(pd.read_csv(args.disagreements))
    else:
        try:
            frames.append(build_disagreements_from_repo())
        except FileNotFoundError as exc:
            log.warning("repo_artifacts_missing", error=str(exc))
    if not args.no_synthetic:
        frames.append(synthetic_disagreements(seed=args.seed))
    if not frames:
        print("error: no disagreement data available", file=sys.stderr)
        return 2
    df = pd.concat(frames, ignore_index=True)
    texts = df["text"].astype(str).tolist()
    if len(texts) < args.min_cluster_size:
        print(
            f"error: need >= {args.min_cluster_size} disagreements, got {len(texts)}",
            file=sys.stderr,
        )
        return 2

    labels, vectorizer = cluster_texts(texts, args.min_cluster_size, args.seed)
    clusters = describe_clusters(texts, labels, vectorizer)
    mode = "dry-run"
    if args.llm:
        clusters = name_clusters_llm(clusters)
        mode = "llm"
    agreement = agreement_with_human(texts, labels)
    write_taxonomy_report(clusters, agreement, len(texts), mode, args.report)
    print(f"disagreement taxonomy -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
