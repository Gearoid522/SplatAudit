"""Thin helpers around the single bundled Graphdeco implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def resolve_model_ply(scene_path: str | Path) -> Path:
    scene = Path(scene_path).resolve()
    if scene.is_file() and scene.suffix.lower() == ".ply":
        return scene
    point_cloud = scene / "point_cloud"
    if not point_cloud.is_dir():
        raise FileNotFoundError(point_cloud)
    iterations = sorted(
        int(path.name.rsplit("_", 1)[-1])
        for path in point_cloud.iterdir()
        if path.is_dir() and path.name.startswith("iteration_")
    )
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directory under {point_cloud}")
    return point_cloud / f"iteration_{iterations[-1]}" / "point_cloud.ply"


def save_selected_ply(model, keep_mask: np.ndarray, path: str | Path) -> Path:
    keep = np.asarray(keep_mask, dtype=bool)
    if keep.ndim != 1 or keep.size != model.get_xyz.shape[0]:
        raise ValueError("keep mask does not match the loaded Gaussian model")
    indices = np.flatnonzero(keep)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    xyz = model._xyz.detach().cpu().numpy()[indices]
    normals = np.zeros_like(xyz)
    features_dc = model._features_dc.detach().cpu().numpy()[indices]
    features_dc = features_dc.transpose(0, 2, 1).reshape(len(indices), -1)
    features_rest = model._features_rest.detach().cpu().numpy()[indices]
    features_rest = features_rest.transpose(0, 2, 1).reshape(len(indices), -1)
    opacities = model._opacity.detach().cpu().numpy()[indices]
    scales = model._scaling.detach().cpu().numpy()[indices]
    rotations = model._rotation.detach().cpu().numpy()[indices]

    attributes = np.concatenate(
        [xyz, normals, features_dc, features_rest, opacities, scales, rotations],
        axis=1,
    )
    dtype = [(name, "f4") for name in model.construct_list_of_attributes()]
    elements = np.empty(len(indices), dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(destination)
    return destination
