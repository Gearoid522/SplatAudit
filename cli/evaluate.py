#!/usr/bin/env python3
"""Evaluate a model on held-out views and render SplatAudit FAS maps."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from splataudit.backends.graphdeco import resolve_model_ply
from splataudit.backends.paths import activate_graphdeco

activate_graphdeco()

from gaussian_renderer import render
from lpipsPyTorch import LPIPS
from scene.cameras import MiniCam
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
from utils.image_utils import psnr
from utils.loss_utils import ssim

from splataudit.geometry.colmap_utils import load_cameras
from splataudit.geometry.load_scene import load_ply
from splataudit.geometry.pipeline import run_pipeline


PIPELINE = Namespace(
    compute_cov3D_python=False,
    convert_SHs_python=False,
    debug=False,
    antialiasing=False,
)


def _camera(camera: dict) -> MiniCam:
    width, height = camera["width"], camera["height"]
    intrinsic = camera["K"]
    fov_x = focal2fov(intrinsic[0, 0], width)
    fov_y = focal2fov(intrinsic[1, 1], height)
    world_view = torch.tensor(
        getWorld2View2(camera["R"].T, camera["t"]), device="cuda"
    ).T
    projection = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y
    ).transpose(0, 1).to("cuda")
    full_projection = world_view.unsqueeze(0).bmm(
        projection.unsqueeze(0)
    ).squeeze(0)
    return MiniCam(
        width=width,
        height=height,
        fovy=fov_y,
        fovx=fov_x,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view,
        full_proj_transform=full_projection,
    )


def _lpips(prediction, target, model, tile_size: int = 1024) -> float:
    height, width = prediction.shape[-2:]
    weighted_sum = 0.0
    total_area = 0
    def intervals(length):
        starts = list(range(0, length, tile_size))
        if len(starts) > 1 and length - starts[-1] < 32:
            starts.pop()
        return [
            (start, starts[index + 1] if index + 1 < len(starts) else length)
            for index, start in enumerate(starts)
        ]

    for y, y_end in intervals(height):
        for x, x_end in intervals(width):
            area = (y_end - y) * (x_end - x)
            weighted_sum += float(
                model(
                    prediction[..., y:y_end, x:x_end],
                    target[..., y:y_end, x:x_end],
                )
            ) * area
            total_area += area
    return weighted_sum / total_area


def _magma(values: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps

    rgb = colormaps["magma"](np.clip(values, 0.0, 1.0))[..., :3]
    return (rgb * 255).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out image metrics and SplatAudit FAS."
    )
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save full-resolution renders and magma FAS maps",
    )
    args = parser.parse_args()

    scene_path = Path(args.scene_path).resolve()
    source_path = Path(args.source_path).resolve()
    output = Path(args.output).resolve()
    split_path = scene_path / "view_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text())
    test_views = set(split["test_views"])
    if test_views & set(split["train_views"]):
        raise ValueError("Training and test views overlap")

    scene = load_ply(str(resolve_model_ply(scene_path)))
    model = scene["_model"]
    scores = run_pipeline(scene)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "scores.npy", scores["C_i"])

    cameras = [
        camera
        for camera in load_cameras(source_path / "sparse" / "0")
        if camera["name"] in test_views
        and (source_path / "images" / camera["name"]).is_file()
    ]
    if len(cameras) != len(test_views):
        found = {camera["name"] for camera in cameras}
        raise ValueError(f"Missing held-out views: {sorted(test_views - found)}")

    if args.save_images:
        render_directory = output / "renders"
        fas_directory = output / "fas"
        render_directory.mkdir()
        fas_directory.mkdir()

    background = torch.zeros(3, device="cuda")
    ones = torch.ones((scene["N"], 3), device="cuda")
    artifact_color = (
        torch.as_tensor(scores["sig_overall"], device="cuda")
        .unsqueeze(1)
        .expand(-1, 3)
        .contiguous()
    )
    lpips_model = LPIPS(net_type="vgg").cuda().eval()
    per_view = {}

    for camera in tqdm(cameras, desc="Evaluating"):
        name = Path(camera["name"]).stem
        view = _camera(camera)
        ground_truth = TF.to_tensor(
            Image.open(source_path / "images" / camera["name"]).convert("RGB")
        ).cuda()

        with torch.no_grad():
            rendered = render(view, model, PIPELINE, background)["render"].clamp(0, 1)
            alpha = render(
                view, model, PIPELINE, background, override_color=ones
            )["render"][0]
            artifact = render(
                view, model, PIPELINE, background, override_color=artifact_color
            )["render"][0]

        normalized = (
            artifact / torch.clamp(alpha, min=1e-8)
        ).clamp(0, 1).cpu().numpy()
        mask = alpha.cpu().numpy() >= 1e-3
        fas = float(normalized[mask].mean()) if mask.any() else 0.0

        with torch.no_grad():
            prediction = rendered.unsqueeze(0)
            target = ground_truth.unsqueeze(0)
            metrics = {
                "PSNR": float(psnr(prediction, target).mean()),
                "SSIM": float(ssim(prediction, target)),
                "LPIPS": _lpips(prediction, target, lpips_model),
                "FAS": fas,
            }
        per_view[name] = {key: round(value, 6) for key, value in metrics.items()}

        if args.save_images:
            render_image = (
                rendered.permute(1, 2, 0).cpu().numpy() * 255
            ).round().astype(np.uint8)
            Image.fromarray(render_image).save(render_directory / f"{name}.png")
            Image.fromarray(_magma(normalized)).save(fas_directory / f"{name}.png")

    averages = {
        key: round(float(np.mean([row[key] for row in per_view.values()])), 6)
        for key in ("PSNR", "SSIM", "LPIPS", "FAS")
    }
    (output / "results.json").write_text(
        json.dumps(
            {
                "scene": str(scene_path),
                "split": str(split_path),
                "primitive_count": scene["N"],
                "per_view": per_view,
                "average": averages,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"PSNR {averages['PSNR']:.4f}  SSIM {averages['SSIM']:.4f}  "
        f"LPIPS {averages['LPIPS']:.4f}  FAS {averages['FAS']:.4f}"
    )


if __name__ == "__main__":
    main()
