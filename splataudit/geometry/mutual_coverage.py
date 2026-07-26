"""Mutual support-coverage signal for each Gaussian primitive."""

from __future__ import annotations

import numpy as np


def compute_mutual_coverage(
    neighbour_counts: np.ndarray, mutual_counts: np.ndarray
) -> np.ndarray:
    """Return ``mutual_counts / (neighbour_counts + 1)``.

    The fixed ``+1`` discounts isolated or very-low-degree primitives: one
    reciprocal edge alone is not treated as complete surface support.
    """
    counts = np.asarray(neighbour_counts, dtype=np.float64)
    mutual = np.asarray(mutual_counts, dtype=np.float64)
    if counts.shape != mutual.shape:
        raise ValueError("neighbour and mutual count arrays must have the same shape")
    if np.any(counts < 0) or np.any(mutual < 0) or np.any(mutual > counts):
        raise ValueError("coverage counts are inconsistent")
    return mutual / (counts + 1.0)
