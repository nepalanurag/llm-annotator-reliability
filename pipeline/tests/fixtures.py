"""Shared synthetic fixtures for the pipeline test suite."""

from __future__ import annotations

import pandas as pd


def fleiss_toy_matrix() -> list[list[int]]:
    """4 subjects x 3 raters x 2 categories, hand-computed kappa = 1/3.

    P_i: 1.0, 1/3, 1/3, 1.0 -> P_bar = 2/3. Marginals 0.5/0.5 -> P_e = 0.5.
    kappa = (2/3 - 1/2) / (1/2) = 1/3.
    """
    return [[3, 0], [2, 1], [1, 2], [0, 3]]


def krippendorff_toy() -> list[list[str]]:
    """3 items x 2 raters, hand-computed nominal alpha = 4/9.

    Coincidences (ordered pairs): o_AA=2, o_BB=2, o_AB=o_BA=1.
    D_o = 2/6 = 1/3; D_e = 18/30 = 0.6; alpha = 1 - (1/3)/0.6 = 4/9.
    """
    return [["A", "A"], ["A", "B"], ["B", "B"]]


def sample_abstracts(n: int = 6) -> pd.DataFrame:
    """Tiny valid RawAbstract frame."""
    return pd.DataFrame(
        {
            "doc_id": [f"pmid:{i}" for i in range(n)],
            "title": ["t"] * n,
            "text": [
                "Insulin resistance and glycemic control in type 2 diabetes patients. "
                "HbA1c was the primary outcome at twelve months."
            ]
            * n,
            "source": ["csv"] * n,
            "retrieved_at": ["2026-10-08T00:00:00+00:00"] * n,
            "ref_label": ["diabetes"] * n,
        }
    )


def near_dup_pair() -> tuple[str, str]:
    """Two long texts differing by one word: Jaccard ~0.9 on 5-grams."""
    base = (
        "In this randomized controlled trial we examined the effect of intensive "
        "blood pressure lowering on cardiovascular outcomes in adults with type "
        "two diabetes and resistant hypertension over twenty four months of "
        "followup with careful monitoring of adverse events and medication "
        "adherence across all participating clinical centers in the study. "
        "The primary endpoint was a composite of myocardial infarction, stroke, "
        "and cardiovascular death, adjudicated by a blinded committee using "
        "standardized criteria applied consistently throughout the trial period."
    )
    variant = base.replace("careful", "rigorous")
    return base, variant
