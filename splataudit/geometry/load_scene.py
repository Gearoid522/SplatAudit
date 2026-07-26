"""
Load a trained 3DGS scene from a .ply file.

Uses GaussianModel.load_ply() from gaussian-splatting directly — the upstream
PLY parser — and build_scaling_rotation from utils/general_utils.py for
covariance reconstruction.  No duplication of parsing or math logic.

Note: GaussianModel.load_ply() and build_scaling_rotation both place tensors
on CUDA (hardcoded in the 3DGS repo).  CUDA is therefore required for loading.
All tensors are converted to CPU numpy after loading for the SplatAudit geometry
pipeline (Stages 1–4).
"""

import math
import numpy as np

from splataudit.backends.paths import activate_graphdeco

activate_graphdeco()

from scene.gaussian_model import GaussianModel
from utils.general_utils import build_scaling_rotation


def load_ply(path: str) -> dict:
    """
    Load a 3DGS .ply file using GaussianModel.load_ply().

    Detects the SH degree automatically from the number of f_rest_ attributes,
    then delegates all PLY parsing to the upstream GaussianModel.

    Returns a dict with:
        positions  : (N, 3)    float64 numpy  — Gaussian centers
        scales     : (N, 3)    float64 numpy  — activated scales (exp applied)
        opacities  : (N,)      float64 numpy  — activated opacities (sigmoid applied)
        sigmas     : (N, 3, 3) float64 numpy  — 3D covariance matrices
        N          : int
    """
    sh_degree = _detect_sh_degree(path)
    model = GaussianModel(sh_degree=sh_degree)
    model.load_ply(path)   # places all tensors on CUDA

    positions  = model.get_xyz.detach().cpu().numpy().astype(np.float64)
    scales     = model.get_scaling.detach().cpu().numpy().astype(np.float64)
    opacities  = model.get_opacity.detach().cpu().numpy().astype(np.float64).squeeze(1)

    # Sigma = L @ L^T  where L = R @ diag(s)
    # Uses build_scaling_rotation from utils/general_utils.py (CUDA tensors in, CUDA tensor out)
    L      = build_scaling_rotation(model.get_scaling, model._rotation)   # (N, 3, 3) CUDA
    sigmas = (L @ L.transpose(1, 2)).detach().cpu().numpy().astype(np.float64)

    return {
        "positions":  positions,
        "scales":     scales,
        "opacities":  opacities,
        "sigmas":     sigmas,
        "N":          len(positions),
        # Keep the upstream GaussianModel for PLY export and rendering.
        "_model": model,
    }


def _detect_sh_degree(path: str) -> int:
    """Infer SH degree from the number of f_rest_ attributes in the .ply file."""
    from plyfile import PlyData
    plydata = PlyData.read(path)
    n_rest = sum(1 for p in plydata.elements[0].properties if p.name.startswith("f_rest_"))
    # 3*(sh_degree+1)^2 - 3 == n_rest  →  sh_degree = sqrt((n_rest+3)/3) - 1
    return int(math.sqrt((n_rest + 3) / 3)) - 1
