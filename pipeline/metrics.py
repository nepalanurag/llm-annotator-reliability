"""Inter-annotator agreement metrics, implemented from first principles.

- fleiss_kappa: Fleiss' kappa for >= 2 raters with a fixed rater count.
- krippendorff_alpha: Krippendorff's alpha, nominal metric, handles missing
  values and a varying rater count per item.
- per_category_agreement: for each category, the fraction of rater pairs that
  agree among pairs where at least one rater chose the category.
- wilson_ci: Wilson score interval for a binomial proportion.

All functions are pure (no I/O) and fully type-hinted. Reference values are
pinned in pipeline/tests/test_metrics.py with hand-computed arithmetic.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Sequence


def fleiss_kappa(count_matrix: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa from an (n_subjects x n_categories) count matrix.

    count_matrix[i][j] = number of raters assigning subject i to category j.
    Every subject must have the same total rater count.
    """
    n = len(count_matrix)
    if n == 0:
        raise ValueError("count_matrix must not be empty")
    n_raters = sum(count_matrix[0])
    if n_raters < 2:
        raise ValueError("need at least 2 raters per subject")
    n_cats = len(count_matrix[0])
    for row in count_matrix:
        if len(row) != n_cats:
            raise ValueError("all rows must have the same category count")
        if sum(row) != n_raters:
            raise ValueError("every subject must have the same rater count")
        if any(c < 0 for c in row):
            raise ValueError("counts must be non-negative")

    # Per-subject agreement: fraction of agreeing rater pairs.
    p_i = [
        sum(c * (c - 1) for c in row) / (n_raters * (n_raters - 1))
        for row in count_matrix
    ]
    p_bar = sum(p_i) / n
    # Chance agreement from marginal category proportions.
    p_j = [sum(row[j] for row in count_matrix) / (n * n_raters) for j in range(n_cats)]
    p_e = sum(p * p for p in p_j)
    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (float(p_bar) - float(p_e)) / (1.0 - float(p_e))


def krippendorff_alpha(
    annotations: Sequence[Sequence[Hashable | None]],
) -> float:
    """Krippendorff's alpha with the nominal (identity) distance metric.

    annotations[i] = labels assigned to item i by any number of raters;
    None marks a missing annotation. Items with fewer than two non-missing
    values are not pairable and are skipped.
    """
    # Collect pairable values per item.
    pairable: list[list[Hashable]] = [
        [v for v in item if v is not None] for item in annotations
    ]
    pairable = [vals for vals in pairable if len(vals) >= 2]
    if not pairable:
        raise ValueError("no pairable items (need >= 2 values on at least one item)")

    value_counts: Counter[Hashable] = Counter()
    for vals in pairable:
        value_counts.update(vals)
    n = sum(value_counts.values())

    # Coincidence matrix over ordered pairs, weighted by 1 / (m_u - 1).
    o: Counter[tuple[Hashable, Hashable]] = Counter()
    for vals in pairable:
        m = len(vals)
        weight = 1.0 / (m - 1)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[(vals[a], vals[b])] += weight

    # Observed disagreement: nominal metric is 0 on the diagonal, 1 elsewhere.
    d_o = sum(cnt for (c, k), cnt in o.items() if c != k) / n
    # Expected disagreement from the marginal value distribution.
    d_e = sum(
        value_counts[c] * value_counts[k]
        for c in value_counts
        for k in value_counts
        if c != k
    ) / (n * (n - 1))
    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def per_category_agreement(
    count_matrix: Sequence[Sequence[int]], categories: Sequence[str]
) -> dict[str, float]:
    """Category-specific agreement for each category.

    For category j: (# agreeing rater pairs both choosing j) /
    (# rater pairs where at least one rater chose j), summed over subjects.
    1.0 = every mention of the category is unanimous; low values flag the
    categories where annotators diverge.
    """
    if len(categories) != len(count_matrix[0]):
        raise ValueError("categories must match the matrix width")
    n_raters = sum(count_matrix[0])
    total_pairs = n_raters * (n_raters - 1) / 2
    out: dict[str, float] = {}
    for j, name in enumerate(categories):
        agree = 0.0
        eligible = 0.0
        for row in count_matrix:
            c = row[j]
            agree += c * (c - 1) / 2
            eligible += total_pairs - (n_raters - c) * (n_raters - c - 1) / 2
        out[name] = agree / eligible if eligible > 0 else float("nan")
    return out


# Exact two-sided 95% normal quantile (not the 1.96 shorthand): this matches the
# repo's published Wilson intervals bit-for-bit (see test_wilson_ci_matches_repo_value).
_Z_95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def count_matrix_from_labels(
    annotations: Sequence[Sequence[Hashable | None]],
    categories: Sequence[str],
) -> list[list[int]]:
    """Build a Fleiss count matrix from per-item rater label lists."""
    matrix: list[list[int]] = []
    for item in annotations:
        counts = [0] * len(categories)
        for v in item:
            if v is not None:
                if v not in categories:
                    raise ValueError(f"label {v!r} not in categories")
                counts[categories.index(v)] += 1
        matrix.append(counts)
    return matrix
