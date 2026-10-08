"""Tests for pipeline.metrics: hand-verified reference values."""

import pytest

from pipeline.metrics import (
    count_matrix_from_labels,
    fleiss_kappa,
    krippendorff_alpha,
    per_category_agreement,
    wilson_ci,
)
from pipeline.tests.fixtures import fleiss_toy_matrix, krippendorff_toy


def test_fleiss_kappa_hand_computed():
    # 4 subjects x 3 raters: kappa = 1/3 by hand (see fixtures.py).
    assert fleiss_kappa(fleiss_toy_matrix()) == pytest.approx(1 / 3, abs=1e-12)


def test_fleiss_kappa_perfect_agreement():
    assert fleiss_kappa([[3, 0], [3, 0], [0, 3]]) == pytest.approx(1.0, abs=1e-12)


def test_fleiss_kappa_rejects_ragged():
    with pytest.raises(ValueError):
        fleiss_kappa([[3, 0], [2, 2]])  # unequal rater totals


def test_fleiss_kappa_rejects_single_rater():
    with pytest.raises(ValueError):
        fleiss_kappa([[1, 0], [0, 1]])


def test_krippendorff_alpha_hand_computed():
    # Nominal alpha = 4/9 by hand (see fixtures.py).
    assert krippendorff_alpha(krippendorff_toy()) == pytest.approx(4 / 9, abs=1e-9)


def test_krippendorff_alpha_perfect():
    assert krippendorff_alpha([["a", "a"], ["b", "b", "b"]]) == pytest.approx(1.0)


def test_krippendorff_alpha_ignores_missing():
    # None values are not pairable; remaining pairs agree perfectly.
    assert krippendorff_alpha([["a", "a", None], ["b", None, "b"]]) == pytest.approx(
        1.0
    )


def test_krippendorff_alpha_needs_pairable():
    with pytest.raises(ValueError):
        krippendorff_alpha([["a"], ["b"], [None, None]])


def test_per_category_agreement_toy():
    # Hand: category A -> 4 agreeing pairs / 8 eligible = 0.5; same for B.
    result = per_category_agreement(fleiss_toy_matrix(), ["A", "B"])
    assert result["A"] == pytest.approx(0.5, abs=1e-12)
    assert result["B"] == pytest.approx(0.5, abs=1e-12)


def test_per_category_agreement_flags_weak_category():
    # Category B is never agreed on: rows pair raters on A or split, never on B.
    matrix = [[2, 0], [2, 0], [1, 1], [1, 1]]
    result = per_category_agreement(matrix, ["A", "B"])
    assert result["A"] == pytest.approx(0.5, abs=1e-12)
    assert result["B"] == pytest.approx(0.0, abs=1e-12)
    assert result["A"] > result["B"]


def test_wilson_ci_matches_repo_value():
    # Cross-check against the repo's published, verified Wilson interval:
    # results.json accuracy 238/248 -> [0.9273789323964281, 0.9779525450642839].
    lo, hi = wilson_ci(238, 248)
    assert lo == pytest.approx(0.9273789323964281, abs=1e-12)
    assert hi == pytest.approx(0.9779525450642839, abs=1e-12)


def test_wilson_ci_edge_cases():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson_ci(10, 10)
    assert hi == pytest.approx(1.0) and lo < 1.0
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_count_matrix_from_labels():
    matrix = count_matrix_from_labels(
        [["a", "a", "b"], ["b", "b", "b"], ["a", None, "a"]], ["a", "b"]
    )
    assert matrix == [[2, 1], [0, 3], [2, 0]]
    with pytest.raises(ValueError):
        count_matrix_from_labels([["zzz"]], ["a", "b"])


def test_kappa_and_alpha_two_rater_values():
    # Two raters, no missing data. Hand computation:
    # Fleiss: counts [2,0],[1,1],[0,2],[1,1],[1,1]; P_bar=2/5, P_e=1/2
    #   -> kappa = (0.4-0.5)/0.5 = -0.2 (Scott's pi).
    # Krippendorff: D_o=6/10, D_e=50/90 (Bessel-corrected marginals)
    #   -> alpha = 1 - 0.6/(50/90) = -0.08.
    # They differ only by Krippendorff's n-1 correction in D_e.
    labels = [["a", "a"], ["a", "b"], ["b", "b"], ["a", "b"], ["b", "a"]]
    matrix = count_matrix_from_labels(labels, ["a", "b"])
    assert fleiss_kappa(matrix) == pytest.approx(-0.2, abs=1e-12)
    assert krippendorff_alpha(labels) == pytest.approx(-0.08, abs=1e-12)
