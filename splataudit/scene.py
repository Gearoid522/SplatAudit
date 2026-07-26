"""Create filtered COLMAP scenes through the shared upstream loaders."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path
from typing import Sequence

from splataudit.backends.paths import load_colmap_loader
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _load_colmap(sparse_dir: str | Path):
    loader = load_colmap_loader()
    root = Path(sparse_dir)
    if (root / "cameras.bin").exists():
        cameras = loader.read_intrinsics_binary(str(root / "cameras.bin"))
        images = loader.read_extrinsics_binary(str(root / "images.bin"))
    else:
        cameras = loader.read_intrinsics_text(str(root / "cameras.txt"))
        images = loader.read_extrinsics_text(str(root / "images.txt"))
    return cameras, images


def colmap_image_names(dataset_dir: str | Path) -> list[str]:
    dataset = Path(dataset_dir)
    _, images = _load_colmap(dataset / "sparse" / "0")
    available = {
        path.name
        for path in (dataset / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    names = sorted(image.name for image in images.values() if image.name in available)
    if not names:
        raise ValueError(f"No COLMAP cameras have matching images under {dataset}")
    return names


def write_filtered_colmap(
    source_sparse: str | Path,
    destination_sparse: str | Path,
    selected_names: Sequence[str],
) -> None:
    """Write a camera model containing exactly ``selected_names``.

    Parsing and camera model definitions come from the bundled 3DGS loader.
    Only the mechanical filtered COLMAP serialization lives here.
    """
    loader = load_colmap_loader()
    source = Path(source_sparse)
    destination = Path(destination_sparse)
    destination.mkdir(parents=True, exist_ok=True)
    cameras, images = _load_colmap(source)
    selected = set(selected_names)
    selected_images = {
        image_id: image for image_id, image in images.items() if image.name in selected
    }
    found = {image.name for image in selected_images.values()}
    missing = sorted(selected - found)
    if missing:
        raise ValueError(f"Selected images missing from COLMAP model: {missing}")

    used_camera_ids = {image.camera_id for image in selected_images.values()}
    with (destination / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(used_camera_ids)))
        for camera_id in sorted(used_camera_ids):
            camera = cameras[camera_id]
            model_id = loader.CAMERA_MODEL_NAMES[camera.model].model_id
            handle.write(
                struct.pack(
                    "<iiQQ", camera_id, model_id, camera.width, camera.height
                )
            )
            handle.write(struct.pack(f"<{len(camera.params)}d", *camera.params))

    with (destination / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(selected_images)))
        for image_id in sorted(selected_images):
            image = selected_images[image_id]
            handle.write(struct.pack("<i", image_id))
            handle.write(struct.pack("<4d", *image.qvec))
            handle.write(struct.pack("<3d", *image.tvec))
            handle.write(struct.pack("<i", image.camera_id))
            handle.write(image.name.encode("utf-8") + b"\x00")
            # Training only requires camera poses, not the original 2D tracks.
            handle.write(struct.pack("<Q", 0))

    for filename in ("points3D.bin", "points3D.txt", "points3D.ply"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, destination / filename)
            break


def materialize_filtered_scene(
    dataset_dir: str | Path,
    destination_dir: str | Path,
    selected_names: Sequence[str],
) -> Path:
    dataset = Path(dataset_dir).resolve()
    destination = Path(destination_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty scene: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    write_filtered_colmap(
        dataset / "sparse" / "0", destination / "sparse" / "0", selected_names
    )
    images_destination = destination / "images"
    images_destination.mkdir()
    for name in selected_names:
        source = dataset / "images" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = images_destination / name
        target.symlink_to(source)
    return destination
