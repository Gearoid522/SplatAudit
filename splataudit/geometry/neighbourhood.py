"""Exact 3-sigma support neighbourhood with bounded working memory.

The production path keeps valid directed edges in compact ``int32`` blocks.
Counts and reciprocal coverage are accumulated while edges are generated, so
the scorer does not need to globally encode, sort, and search the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def _sym3_inv_compact(matrices: np.ndarray) -> np.ndarray:
    """Invert symmetric 3x3 matrices into their six unique coefficients."""
    matrices = np.asarray(matrices, dtype=np.float32)
    matrix_scale = np.maximum(
        np.trace(matrices, axis1=1, axis2=2) / 3.0,
        np.finfo(np.float32).tiny,
    )
    eps = (1e-6 * matrix_scale).astype(np.float32, copy=False)
    a = matrices[:, 0, 0] + eps
    b = matrices[:, 0, 1]
    c = matrices[:, 0, 2]
    d = matrices[:, 1, 1] + eps
    e = matrices[:, 1, 2]
    f = matrices[:, 2, 2] + eps

    c00 = d * f - e * e
    c01 = c * e - b * f
    c02 = b * e - c * d
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    determinant = a * c00 + b * c01 + c * c02
    inverse_determinant = 1.0 / (determinant + 1e-30)

    return np.column_stack(
        (
            c00 * inverse_determinant,
            c01 * inverse_determinant,
            c02 * inverse_determinant,
            c11 * inverse_determinant,
            c12 * inverse_determinant,
            c22 * inverse_determinant,
        )
    ).astype(np.float32, copy=False)


def _quadratic_form(
    differences: np.ndarray, inverse_compact: np.ndarray, matrix_indices: np.ndarray
) -> np.ndarray:
    """Evaluate ``d.T @ inverse[matrix_indices] @ d`` without 3x3 gathers."""
    inv = inverse_compact[matrix_indices]
    x, y, z = differences.T
    return (
        inv[:, 0] * x * x
        + 2.0 * inv[:, 1] * x * y
        + 2.0 * inv[:, 2] * x * z
        + inv[:, 3] * y * y
        + 2.0 * inv[:, 4] * y * z
        + inv[:, 5] * z * z
    )


@dataclass(frozen=True)
class EdgeBlocks:
    """Compact directed edges ``covering j in neighbourhood N_i``."""

    covered_blocks: tuple[np.ndarray, ...]
    covering_run_blocks: tuple[np.ndarray, ...]
    covering_run_length_blocks: tuple[np.ndarray, ...]
    counts: np.ndarray
    mutual_counts: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(sum(block.size for block in self.covered_blocks))

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for covered, covering_runs, run_lengths in self.iter_compressed():
            yield covered, np.repeat(covering_runs, run_lengths)

    def iter_compressed(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return iter(
            zip(
                self.covered_blocks,
                self.covering_run_blocks,
                self.covering_run_length_blocks,
            )
        )

def _run_length_encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode a nondecreasing int32 array without changing edge order."""
    if values.size == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    if np.any(values[1:] < values[:-1]):
        raise RuntimeError("covering indices must be nondecreasing")
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(values[1:] != values[:-1]).astype(np.int64) + 1,
        )
    )
    lengths = np.diff(
        np.append(starts, np.array([values.size], dtype=np.int64))
    )
    return (
        values[starts].astype(np.int32, copy=False),
        lengths.astype(np.int32, copy=False),
    )


class _TorchEdgeTester:
    """Optional CUDA implementation of exact per-candidate predicates."""

    def __init__(
        self,
        positions: np.ndarray,
        inverse_compact: np.ndarray,
        radii: np.ndarray,
        device: str,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Torch edge testing requested but torch is unavailable"
            ) from error
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA edge testing requested but CUDA is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.positions = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device
        )
        self.inverse = torch.as_tensor(
            inverse_compact, dtype=torch.float32, device=self.device
        )
        self.radii = torch.as_tensor(
            radii, dtype=torch.float32, device=self.device
        )

    def _quadratic(self, differences, matrix_indices):
        inverse = self.inverse[matrix_indices]
        x, y, z = differences.unbind(dim=1)
        return (
            inverse[:, 0] * x * x
            + 2.0 * inverse[:, 1] * x * y
            + 2.0 * inverse[:, 2] * x * z
            + inverse[:, 3] * y * y
            + 2.0 * inverse[:, 4] * y * z
            + inverse[:, 5] * z * z
        )

    def filter(
        self, covered: np.ndarray, covering: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        query = torch.as_tensor(
            covered.astype(np.int64, copy=False), device=self.device
        )
        neighbour = torch.as_tensor(
            covering.astype(np.int64, copy=False), device=self.device
        )
        differences = self.positions[query] - self.positions[neighbour]
        forward = self._quadratic(differences, neighbour) < 9.0
        query = query[forward]
        neighbour = neighbour[forward]
        differences = differences[forward]
        reverse_sphere = torch.sum(differences * differences, dim=1) <= (
            self.radii[query] * self.radii[query]
        )
        reciprocal = reverse_sphere & (self._quadratic(differences, query) < 9.0)
        return (
            query.to("cpu", dtype=torch.int32).numpy(),
            neighbour.to("cpu", dtype=torch.int32).numpy(),
            reciprocal.to("cpu").numpy(),
        )


def _filter_edge_batch_cpu(
    positions: np.ndarray,
    inverse_matrices: np.ndarray,
    radii: np.ndarray,
    covered: np.ndarray,
    covering: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences = positions[covered] - positions[covering]
    forward_distance = _quadratic_form(
        differences, inverse_matrices, covering
    )
    forward = forward_distance < 9.0
    valid_covered = covered[forward].astype(np.int32, copy=False)
    valid_covering = covering[forward].astype(np.int32, copy=False)
    valid_differences = differences[forward]
    reverse_sphere = np.einsum(
        "ij,ij->i", valid_differences, valid_differences
    ) <= np.square(radii[valid_covered])
    reverse_distance = _quadratic_form(
        valid_differences, inverse_matrices, valid_covered
    )
    reciprocal = reverse_sphere & (
        reverse_distance < 9.0
    )
    return valid_covered, valid_covering, reciprocal


def build_3sigma_edge_blocks(
    positions: np.ndarray,
    scales: np.ndarray,
    sigmas: np.ndarray,
    *,
    show_progress: bool = True,
    edge_test_device: str = "auto",
) -> EdgeBlocks:
    """Build the exact directed 3-sigma graph as compact streamed blocks.

    A forward edge is accepted iff
    ``(mu_i-mu_j).T @ inv(Sigma_j) @ (mu_i-mu_j) < 9``.
    Its reciprocal status is evaluated directly with ``Sigma_i``; this is
    identical to searching for the reverse edge after graph construction.
    """
    if edge_test_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("edge_test_device must be cpu, cuda, or auto")
    positions32 = np.asarray(positions, dtype=np.float32)
    scales32 = np.asarray(scales, dtype=np.float32)
    sigmas32 = np.asarray(sigmas, dtype=np.float32)
    n = len(positions32)
    if positions32.shape != (n, 3) or scales32.shape != (n, 3):
        raise ValueError("positions and scales must have shape (N, 3)")
    if sigmas32.shape != (n, 3, 3):
        raise ValueError("sigmas must have shape (N, 3, 3)")
    if n > np.iinfo(np.int32).max:
        raise ValueError("compact graph supports at most int32-indexed primitives")

    resolved_device = edge_test_device
    if resolved_device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"
    inverse_matrices = _sym3_inv_compact(sigmas32)
    radii = 3.0 * scales32.max(axis=1)
    tree = cKDTree(positions32)
    torch_tester = (
        _TorchEdgeTester(
            positions32,
            inverse_matrices,
            radii,
            resolved_device,
        )
        if resolved_device != "cpu"
        else None
    )

    counts = np.zeros(n, dtype=np.int64)
    mutual_counts = np.zeros(n, dtype=np.int64)
    covered_blocks: list[np.ndarray] = []
    covering_run_blocks: list[np.ndarray] = []
    covering_run_length_blocks: list[np.ndarray] = []
    chunk_size = 100_000
    mahalanobis_size = 2_000_000
    chunk_starts = range(0, n, chunk_size)

    iterator = tqdm(
        chunk_starts,
        total=(n + chunk_size - 1) // chunk_size,
        desc="  exact 3σ edges",
        unit="chunk",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for start in iterator:
        end = min(start + chunk_size, n)
        results = tree.query_ball_point(
            positions32[start:end],
            radii[start:end],
            workers=-1,
            return_sorted=True,
        )
        lengths = np.fromiter(
            (len(result) for result in results), np.int64, end - start
        )
        if not lengths.any():
            continue

        covering = np.repeat(
            np.arange(start, end, dtype=np.int32), lengths
        )
        covered = np.concatenate(results, dtype=np.int32)
        not_self = covered != covering
        covered = covered[not_self]
        covering = covering[not_self]
        if covered.size == 0:
            continue

        chunk_covered: list[np.ndarray] = []
        chunk_mutual: list[np.ndarray] = []
        for batch_start in range(0, covered.size, mahalanobis_size):
            batch_end = min(batch_start + mahalanobis_size, covered.size)
            batch = slice(batch_start, batch_end)
            batch_covered = covered[batch]
            batch_covering = covering[batch]
            if torch_tester is None:
                (
                    valid_covered,
                    valid_covering,
                    reciprocal,
                ) = _filter_edge_batch_cpu(
                    positions32,
                    inverse_matrices,
                    radii,
                    batch_covered,
                    batch_covering,
                )
            else:
                (
                    valid_covered,
                    valid_covering,
                    reciprocal,
                ) = torch_tester.filter(batch_covered, batch_covering)
            if valid_covered.size == 0:
                continue

            # The edge relation is unchanged; only its elementwise execution
            # backend differs.
            if reciprocal.shape != valid_covered.shape:
                raise RuntimeError("reciprocal mask shape does not match valid edges")
            covering_runs, covering_run_lengths = _run_length_encode(
                valid_covering
            )
            covered_blocks.append(valid_covered)
            covering_run_blocks.append(covering_runs)
            covering_run_length_blocks.append(covering_run_lengths)
            chunk_covered.append(valid_covered)
            if reciprocal.any():
                chunk_mutual.append(valid_covered[reciprocal])
        if chunk_covered:
            query_indices = np.concatenate(chunk_covered)
            counts += np.bincount(query_indices, minlength=n)
        if chunk_mutual:
            mutual_indices = np.concatenate(chunk_mutual)
            mutual_counts += np.bincount(mutual_indices, minlength=n)
    return EdgeBlocks(
        tuple(covered_blocks),
        tuple(covering_run_blocks),
        tuple(covering_run_length_blocks),
        counts,
        mutual_counts,
    )
