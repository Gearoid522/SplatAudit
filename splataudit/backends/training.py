"""Shared adapter for the pinned Graphdeco training implementation."""

from __future__ import annotations

import argparse
import importlib.util
import random
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator, Sequence

import numpy as np
import torch

from splataudit.backends.paths import activate_graphdeco

GRAPHDECO_ROOT = activate_graphdeco()

# These imports intentionally follow activate_graphdeco(), which places the
# single pinned backend on sys.path.
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def _load_training_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "splataudit_graphdeco_train", GRAPHDECO_ROOT / "train.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Graphdeco training module from {GRAPHDECO_ROOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING_MODULE = _load_training_module()


def parse_training_args(
    source: str | Path,
    output: str | Path,
    extra_args: Sequence[str],
    *,
    default_iterations: int | None = None,
    disable_densification: bool = False,
):
    """Parse the upstream arguments while fixing source and output paths."""
    parser = argparse.ArgumentParser(description="3DGS training parameters")
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(
        debug_from=-1,
        detect_anomaly=False,
        test_iterations=[],
        save_iterations=[],
        checkpoint_iterations=[],
        start_checkpoint=None,
    )
    if default_iterations is not None:
        parser.set_defaults(iterations=default_iterations)

    argv = ["-s", str(source), "-m", str(output), *extra_args]
    args = parser.parse_args(argv)
    if disable_densification:
        args.densify_until_iter = 0
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    return args, model_params, optimization_params, pipeline_params


@contextmanager
def initialize_from_ply(ply_path: str | Path) -> Iterator[None]:
    """Temporarily replace the backend's initial geometry with a PLY."""
    source = str(Path(ply_path).resolve())
    original_setup = TRAINING_MODULE.GaussianModel.training_setup

    def patched_setup(model, options):
        model.load_ply(source)
        model.max_radii2D = torch.zeros(model.get_xyz.shape[0], device="cuda")
        print(f"[refine] Loaded initial PLY: {model.get_xyz.shape[0]:,} primitives")
        original_setup(model, options)

    TRAINING_MODULE.GaussianModel.training_setup = patched_setup
    try:
        yield
    finally:
        TRAINING_MODULE.GaussianModel.training_setup = original_setup


def run_training(
    args,
    model_params,
    optimization_params,
    pipeline_params,
    *,
    seed: int = 0,
) -> None:
    """Run the pinned backend with deterministic initialization state."""
    safe_state(args.quiet)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    TRAINING_MODULE.training(
        model_params.extract(args),
        optimization_params.extract(args),
        pipeline_params.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
