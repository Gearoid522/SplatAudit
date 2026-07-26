"""Final SplatAudit hybrid pruning score."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _zscore(values: np.ndarray) -> np.ndarray:
    deviation = values.std()
    if deviation == 0.0:
        return np.zeros_like(values)
    return (values - values.mean()) / deviation


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1.0)


def hybrid_score(
    confidence: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Fuse SplatAudit confidence with PointSplat intrinsic importance.

    PointSplat importance follows its published 0.3 opacity and 0.7 volume
    weighting. Its percentile rank and the percentile rank of SplatAudit
    confidence are combined by their harmonic mean.
    """
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    opacities = np.asarray(opacities, dtype=np.float64).reshape(-1)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (confidence.size, 3) or opacities.shape != confidence.shape:
        raise ValueError("confidence, opacity, and scale counts must match")
    if not np.all(np.isfinite(confidence)) or not np.all(np.isfinite(opacities)):
        raise ValueError("hybrid inputs must be finite")

    volume = (4.0 / 3.0) * np.pi * np.prod(scales, axis=1)
    importance = 0.3 * _zscore(opacities) + 0.7 * _zscore(volume)
    appearance_rank = _percentile_rank(importance)
    geometry_rank = _percentile_rank(confidence)
    denominator = appearance_rank + geometry_rank
    return np.divide(
        2.0 * appearance_rank * geometry_rank,
        denominator,
        out=np.zeros_like(confidence),
        where=denominator > 0.0,
    )
