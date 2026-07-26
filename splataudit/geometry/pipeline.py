"""SplatAudit confidence computation."""

from __future__ import annotations

import numpy as np

from .mutual_coverage import compute_mutual_coverage
from .neighbourhood import build_3sigma_edge_blocks
from .surface_variation import compute_surface_variation_streamed


def run_pipeline(
    scene: dict,
    *,
    verbose: bool = True,
    edge_test_device: str = "auto",
) -> dict[str, np.ndarray]:
    """Return primitive confidence and its projected artifact signal."""
    if verbose:
        print("Building the 3σ support graph...")
    edges = build_3sigma_edge_blocks(
        scene["positions"],
        scene["scales"],
        scene["sigmas"],
        show_progress=verbose,
        edge_test_device=edge_test_device,
    )
    variation = compute_surface_variation_streamed(
        scene["positions"], scene["opacities"], edges
    )
    normalized_variation = np.clip(3.0 * variation, 0.0, 1.0)
    coverage = compute_mutual_coverage(edges.counts, edges.mutual_counts)
    confidence = (1.0 - normalized_variation) * coverage
    confidence[edges.counts < 4] = 0.0
    return {
        "C_i": confidence,
        "sig_overall": (1.0 - confidence).astype(np.float32),
    }
