"""
Stage 2: Surface Variation (v_i)

For each primitive g_i, compute the planarity of its neighbourhood N_i, then
assign that planarity score to each covering primitive g_j ∈ N_i via a mean.

    s_i = λ_min / trace(C_i)    ∈ [0, 1/3]

    v_j = mean { s_i : i such that j ∈ N_i, |N_i| ≥ 4 }

where C_i is the opacity-weighted covariance of neighbour centres {μ_j : j ∈ N_i}.

Interpretation:
  v_j ≈ 0:    g_j consistently appears in planar neighbourhoods → surface primitive
  v_j ≈ 1/3:  g_j consistently appears in isotropic neighbourhoods → floater

Why assign to g_j, not g_i:
  The planarity of N_i is a property of the covering primitives g_j (their positions
  form C_i), not of g_i. Assigning to g_j means g_j's out-of-plane position raises
  the planarity of every N_i it contributes to — g_j accumulates those elevated
  scores rather than the target g_i getting an undeserved low score.

Edge cases:
  - |N_i| < 4: not enough neighbours for a full-rank covariance → s_i not computed,
    contributes no score to any g_j.
  - g_j never appears in any valid N_i: v_j defaults to 1/3 (worst-case isotropic),
    correctly flagging it as an isolated primitive.

Returns:
    v_i        : (N,) float64  surface variation in [0, 1/3]
    local_covs : (N, 3, 3) float64  C_i indexed by query primitive i
"""

import numpy as np

from .neighbourhood import EdgeBlocks


def _accumulate_at(
    destination: np.ndarray, indices: np.ndarray, values: np.ndarray
) -> None:
    """Deterministic repeated-index accumulation."""
    np.add.at(destination, indices, values)


def compute_surface_variation_streamed(
    positions: np.ndarray,
    opacities: np.ndarray,
    edges: EdgeBlocks,
) -> np.ndarray:
    """Compute the unchanged assign-to-coverer score from compact edge blocks.

    The implementation performs bounded-memory passes over the directed graph
    instead of materializing several arrays with one row per edge.
    """
    positions = np.asarray(positions, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64).reshape(-1)
    n = len(positions)
    if positions.shape != (n, 3) or opacities.shape != (n,):
        raise ValueError("positions must be (N, 3) and opacities must be (N,)")
    if len(edges.counts) != n:
        raise ValueError("edge counts do not match primitive count")

    variation = np.full(n, 1.0 / 3.0, dtype=np.float64)
    valid_queries = edges.counts >= 4
    if edges.edge_count == 0 or not valid_queries.any():
        return variation

    # Pass 1: opacity sums and weighted first moments.
    weight_sum = np.zeros(n, dtype=np.float64)
    first_moment = np.zeros((n, 3), dtype=np.float64)
    for covered, covering in edges:
        valid = valid_queries[covered]
        if not valid.any():
            continue
        query = covered[valid]
        neighbour = covering[valid]
        weights = opacities[neighbour]
        _accumulate_at(weight_sum, query, weights)
        neighbour_positions = positions[neighbour]
        for axis in range(3):
            _accumulate_at(
                first_moment[:, axis],
                query,
                weights * neighbour_positions[:, axis],
            )

    means = first_moment / (weight_sum[:, None] + 1e-8)
    del first_moment

    # Pass 2: six unique entries of the centered covariance.
    covariance = np.zeros((n, 6), dtype=np.float64)
    for covered, covering in edges:
        valid = valid_queries[covered]
        if not valid.any():
            continue
        query = covered[valid]
        neighbour = covering[valid]
        weights = opacities[neighbour] / (weight_sum[query] + 1e-8)
        centered = positions[neighbour] - means[query]
        x, y, z = centered.T
        products = (x * x, x * y, x * z, y * y, y * z, z * z)
        for column, product in enumerate(products):
            _accumulate_at(covariance[:, column], query, weights * product)
    del means, weight_sum

    # Batched eigendecomposition bounds the temporary (V, 3, 3) allocation.
    planarity = np.empty(n, dtype=np.float64)
    valid_indices = np.flatnonzero(valid_queries)
    batch_size = 250_000
    for start in range(0, len(valid_indices), batch_size):
        indices = valid_indices[start : start + batch_size]
        compact = covariance[indices]
        matrices = np.empty((len(indices), 3, 3), dtype=np.float64)
        matrices[:, 0, 0] = compact[:, 0]
        matrices[:, 0, 1] = matrices[:, 1, 0] = compact[:, 1]
        matrices[:, 0, 2] = matrices[:, 2, 0] = compact[:, 2]
        matrices[:, 1, 1] = compact[:, 3]
        matrices[:, 1, 2] = matrices[:, 2, 1] = compact[:, 4]
        matrices[:, 2, 2] = compact[:, 5]
        eigenvalues = np.clip(np.linalg.eigvalsh(matrices), 0.0, None)
        trace = eigenvalues.sum(axis=1)
        planarity[indices] = np.divide(
            eigenvalues[:, 0],
            trace,
            out=np.full(len(indices), 1.0 / 3.0, dtype=np.float64),
            where=trace > 1e-12,
        )
    del covariance

    # Pass 3: assign each query's planarity to every covering primitive.
    variation_sum = np.zeros(n, dtype=np.float64)
    contribution_count = np.zeros(n, dtype=np.int64)
    for covered, covering_runs, run_lengths in edges.iter_compressed():
        valid = valid_queries[covered]
        starts = np.empty(len(run_lengths), dtype=np.int64)
        starts[0] = 0
        if len(starts) > 1:
            np.cumsum(run_lengths[:-1], out=starts[1:])
        values = np.where(valid, planarity[covered], 0.0)
        block_sums = np.add.reduceat(values, starts)
        block_counts = np.add.reduceat(valid.astype(np.int64), starts)
        variation_sum[covering_runs] += block_sums
        contribution_count[covering_runs] += block_counts

    has_contribution = contribution_count > 0
    variation[has_contribution] = (
        variation_sum[has_contribution] / contribution_count[has_contribution]
    )
    return variation
