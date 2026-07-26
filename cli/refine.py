#!/usr/bin/env python3
"""Prune by SplatAudit confidence and refine without densification."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from splataudit.backends.graphdeco import resolve_model_ply, save_selected_ply
from splataudit.geometry.hybrid import hybrid_score
from splataudit.geometry.load_scene import load_ply
from splataudit.geometry.pipeline import run_pipeline
from splataudit.scene import materialize_filtered_scene


def _load_split(scene: Path) -> dict:
    split_path = scene / "view_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"View split not found: {split_path}. "
            "Use a model created by cli/train.py."
        )
    split = json.loads(split_path.read_text())
    if set(split["train_views"]) & set(split["test_views"]):
        raise ValueError("Training and test views overlap")
    return split


def _scores(args, scene_data: dict):
    scores = run_pipeline(scene_data, verbose=True)["C_i"]
    if args.method == "hybrid":
        scores = hybrid_score(
            scores, scene_data["opacities"], scene_data["scales"]
        )
    return scores


def _select(args, scores):
    scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=-np.inf,
        neginf=-np.inf,
        posinf=np.inf,
    )
    if args.prune_below is not None:
        return "threshold", scores >= args.prune_below

    count = math.floor(scores.size * args.keep + 0.5)
    indices = np.arange(scores.size)
    ranking = np.lexsort((indices, -scores))
    if args.selection_policy == "strict-boundary":
        boundary = scores[ranking[count - 1]] if count else np.inf
        return "strict-boundary", scores > boundary
    keep = np.zeros(scores.size, dtype=bool)
    keep[ranking[:count]] = True
    return "exact-count", keep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune a 3DGS model with SplatAudit and refine it."
    )
    parser.add_argument("--scene-path", "--scene_path", dest="scene_path", required=True)
    parser.add_argument("--train-path", "--train_path", dest="train_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=("splataudit", "hybrid"),
        default="splataudit",
    )
    parser.add_argument("--keep", type=float, default=None)
    parser.add_argument(
        "--prune-below", "--prune_below", dest="prune_below", type=float, default=None
    )
    parser.add_argument(
        "--selection-policy",
        choices=("exact-count", "strict-boundary"),
        default="exact-count",
        help=(
            "exact-count keeps the requested budget; strict-boundary removes all "
            "scores at or below the boundary"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args, upstream_args = parser.parse_known_args()
    from splataudit.backends.training import (
        initialize_from_ply,
        parse_training_args,
        run_training,
    )

    if sum((args.keep is not None, args.prune_below is not None)) != 1:
        parser.error("choose exactly one of --keep or --prune-below")
    if args.selection_policy != "exact-count" and args.keep is None:
        parser.error("--selection-policy applies only with --keep")
    if args.keep is not None and not 0.0 < args.keep <= 1.0:
        parser.error("--keep must be in (0, 1]")
    if args.prune_below is not None and not 0.0 <= args.prune_below <= 1.0:
        parser.error("--prune-below must be in [0, 1]")

    scene_path = Path(args.scene_path).resolve()
    dataset = Path(args.train_path).resolve()
    output = Path(args.output).resolve()
    selection_scene: str | None = None
    training_scene: str | None = None

    try:
        input_ply = resolve_model_ply(scene_path)
        print(f"[refine] Loading {input_ply}")
        scene_data = load_ply(str(input_ply))
        primitive_count = scene_data["N"]
        model = scene_data["_model"]

        scores = _scores(args, scene_data)
        policy, keep_mask = _select(args, scores)
        kept_count = int(keep_mask.sum())
        print(
            f"[refine] {args.method}/{policy}: "
            f"{kept_count:,} / {primitive_count:,} kept"
        )
        selection_scene = tempfile.mkdtemp(prefix="splataudit_selection_")
        selected_ply = (
            Path(selection_scene) / "point_cloud" / "iteration_1" / "point_cloud.ply"
        )
        save_selected_ply(model, keep_mask, selected_ply)

        scene_data["_model"] = None
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        split = _load_split(scene_path)
        training_scene = tempfile.mkdtemp(prefix="splataudit_train_")
        train_source = materialize_filtered_scene(
            dataset, training_scene, split["train_views"]
        )

        training_args, model_params, optimization_params, pipeline_params = (
            parse_training_args(
                train_source,
                output,
                upstream_args,
                default_iterations=5_000,
                disable_densification=True,
            )
        )
        print(f"[refine] Refining for {training_args.iterations} iterations")
        with initialize_from_ply(selected_ply):
            run_training(
                training_args,
                model_params,
                optimization_params,
                pipeline_params,
                seed=args.seed,
            )
        gc.collect()
        torch.cuda.empty_cache()
        output.mkdir(parents=True, exist_ok=True)
        (output / "view_split.json").write_text(json.dumps(split, indent=2) + "\n")

    finally:
        if selection_scene is not None:
            shutil.rmtree(selection_scene, ignore_errors=True)
        if training_scene is not None:
            shutil.rmtree(training_scene, ignore_errors=True)


if __name__ == "__main__":
    main()
