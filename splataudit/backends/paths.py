"""Resolve bundled backends without copying their implementation."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPHDECO_ROOT = PROJECT_ROOT / "third_party" / "gaussian-splatting"


def graphdeco_root() -> Path:
    path = GRAPHDECO_ROOT.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Graphdeco submodule not found: {path}. "
            "Run git submodule update --init --recursive."
        )
    return path


def activate_graphdeco() -> Path:
    """Put the shared 3DGS implementation first on ``sys.path``."""
    root = graphdeco_root()
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    return root


@lru_cache(maxsize=1)
def load_colmap_loader() -> ModuleType:
    """Load only Graphdeco's COLMAP helpers without importing its CUDA stack."""
    source = graphdeco_root() / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location(
        "splataudit_graphdeco_colmap_loader", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Graphdeco COLMAP helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
