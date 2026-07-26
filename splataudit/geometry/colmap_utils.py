"""Camera loading through the pinned Graphdeco COLMAP implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from splataudit.backends.paths import load_colmap_loader


def load_cameras(colmap_dir: str | Path) -> list[dict]:
    loader = load_colmap_loader()
    root = Path(colmap_dir)
    if (root / "cameras.bin").is_file():
        camera_data = loader.read_intrinsics_binary(str(root / "cameras.bin"))
        image_data = loader.read_extrinsics_binary(str(root / "images.bin"))
    else:
        camera_data = loader.read_intrinsics_text(str(root / "cameras.txt"))
        image_data = loader.read_extrinsics_text(str(root / "images.txt"))

    cameras = []
    for image in image_data.values():
        camera = camera_data[image.camera_id]
        cameras.append(
            {
                "name": image.name,
                "R": np.asarray(loader.qvec2rotmat(image.qvec), dtype=np.float64),
                "t": np.asarray(image.tvec, dtype=np.float64),
                "K": _intrinsics(camera),
                "width": int(camera.width),
                "height": int(camera.height),
            }
        )
    return sorted(cameras, key=lambda item: item["name"])


def _intrinsics(camera) -> np.ndarray:
    parameters = camera.params
    shared_focal_models = {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    }
    separate_focal_models = {
        "PINHOLE",
        "OPENCV",
        "FULL_OPENCV",
        "OPENCV_FISHEYE",
    }
    if camera.model in shared_focal_models:
        fx = fy = parameters[0]
        cx, cy = parameters[1:3]
    elif camera.model in separate_focal_models:
        fx, fy, cx, cy = parameters[:4]
    else:
        fx = fy = parameters[0]
        cx = parameters[1] if len(parameters) > 1 else camera.width / 2.0
        cy = parameters[2] if len(parameters) > 2 else camera.height / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
