#!/usr/bin/env python3
"""Train a 3DGS baseline with a deterministic sparse or full view split."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from splataudit.scene import (
    colmap_image_names,
    materialize_filtered_scene,
)


def _even_sample(names: list[str], count: int) -> list[str]:
    if not 0 < count <= len(names):
        raise ValueError(f"Requested {count} views from a pool of {len(names)}")
    if count == len(names):
        return names
    if count == 1:
        return [names[(len(names) - 1) // 2]]
    denominator = count - 1
    span = len(names) - 1
    indices = [
        (position * span + denominator // 2) // denominator
        for position in range(count)
    ]
    return [names[index] for index in indices]


def _split_for_run(args, dataset: Path) -> dict:
    all_views = colmap_image_names(dataset)
    test_views = all_views[::8]
    test_set = set(test_views)
    train_pool = [name for name in all_views if name not in test_set]
    train_views = train_pool if args.regime == "full" else _even_sample(train_pool, 24)
    return {
        "all_views": all_views,
        "train_views": train_views,
        "test_views": test_views,
        "train_count": len(train_views),
        "hold_every": 8,
        "regime": args.regime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the common 3DGS baseline on sparse24 or full views."
    )
    parser.add_argument("--scene-path", "--scene_path", dest="scene_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regime", choices=("sparse24", "full"), default="sparse24")
    parser.add_argument("--seed", type=int, default=0)
    args, upstream_args = parser.parse_known_args()
    # Delay the CUDA-heavy backend import until local arguments are parsed.
    from splataudit.backends.training import (
        parse_training_args,
        run_training,
    )

    dataset = Path(args.scene_path).resolve()
    output = Path(args.output).resolve()
    split = _split_for_run(args, dataset)
    selected = split["train_views"]
    temporary_scene: str | None = None

    try:
        print(
            f"[train] Views: {len(split['train_views'])} train / "
            f"{len(split['test_views'])} held out"
        )
        temporary_scene = tempfile.mkdtemp(prefix="splataudit_train_")
        train_source = materialize_filtered_scene(dataset, temporary_scene, selected)

        training_args, model_params, optimization_params, pipeline_params = (
            parse_training_args(train_source, output, upstream_args)
        )
        print(f"[train] Output: {output}")
        print(f"[train] Iterations: {training_args.iterations}")

        run_training(
            training_args,
            model_params,
            optimization_params,
            pipeline_params,
            seed=args.seed,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "view_split.json").write_text(json.dumps(split, indent=2) + "\n")

    finally:
        if temporary_scene is not None:
            shutil.rmtree(temporary_scene, ignore_errors=True)


if __name__ == "__main__":
    main()
