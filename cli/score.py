#!/usr/bin/env python3
"""Compute per-primitive SplatAudit confidence scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from splataudit.backends.graphdeco import resolve_model_ply
from splataudit.geometry.hybrid import hybrid_score
from splataudit.geometry.load_scene import load_ply
from splataudit.geometry.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SplatAudit confidence for a trained 3DGS model."
    )
    parser.add_argument("--scene", required=True, help="3DGS model directory or PLY")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument(
        "--method",
        choices=("splataudit", "hybrid"),
        default="splataudit",
        help="Pruning score to export (default: splataudit)",
    )
    parser.add_argument(
        "--geometry-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for exact support predicates (default: auto)",
    )
    args = parser.parse_args()

    ply_path = resolve_model_ply(args.scene)
    scene = load_ply(str(ply_path))
    result = run_pipeline(
        scene,
        edge_test_device=args.geometry_device,
    )
    scores = result["C_i"]
    if args.method == "hybrid":
        scores = hybrid_score(scores, scene["opacities"], scene["scales"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, scores)
    print(f"Scores: {output}")


if __name__ == "__main__":
    main()
