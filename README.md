# SplatAudit

SplatAudit is a reference-free geometric audit for trained 3D Gaussian
Splatting representations. It builds an exact directed 3σ support graph,
combines opacity-weighted surface variation with mutual coverage, and returns
a confidence value for every Gaussian. The same confidence supports
image-space FAS diagnostics and one-pass confidence-guided pruning.

## Installation

Install Miniconda, NVIDIA driver support, and the CUDA 11.8 toolkit, then run:

```bash
git clone --recursive https://github.com/Gearoid522/SplatAudit.git
cd SplatAudit
conda env create -f environment.yml
conda activate splataudit
TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-build-isolation -e . third_party/gaussian-splatting/submodules/{diff-gaussian-rasterization,simple-knn,fused-ssim}
```

This creates one `splataudit` environment. Conda packages use Tsinghua TUNA;
PyTorch comes from the official CUDA 11.8 wheel index so that pip cannot
silently install a CPU-only build.

## Workflow

Train a deterministic sparse-24 3DGS model:

```bash
python cli/train.py \
  --scene-path /path/to/mipnerf360/bicycle \
  --seed 0 \
  --output outputs/bicycle/baseline \
  --iterations 30000
```

The default `sparse24` regime deterministically holds out every eighth camera
and selects 24 evenly spaced training views. Use `--regime full` to train on
all remaining cameras.

Optionally export either pruning score for inspection:

```bash
python cli/score.py \
  --scene outputs/bicycle/baseline \
  --method splataudit \
  --output outputs/bicycle/splataudit_scores.npy \
  --geometry-device auto

python cli/score.py \
  --scene outputs/bicycle/baseline \
  --method hybrid \
  --output outputs/bicycle/hybrid_scores.npy \
  --geometry-device auto
```

`splataudit` uses geometric confidence directly. `hybrid` uses the finalized
harmonic fusion of ranked SplatAudit confidence and the published PointSplat
intrinsic score. Both modes perform a single pruning pass.

Prune and refine without densification:

```bash
python cli/refine.py \
  --scene-path outputs/bicycle/baseline \
  --train-path /path/to/mipnerf360/bicycle \
  --method splataudit \
  --keep 0.20 \
  --seed 0 \
  --output outputs/bicycle/splataudit_20 \
  --iterations 5000

python cli/refine.py \
  --scene-path outputs/bicycle/baseline \
  --train-path /path/to/mipnerf360/bicycle \
  --method hybrid \
  --keep 0.10 \
  --seed 0 \
  --output outputs/bicycle/hybrid_10 \
  --iterations 10000
```

`--method` and `--keep` are independent controls: either pruning score can be
used at any retention fraction. Training and refinement default to seed `0`.

`--selection-policy exact-count` is the default. Use
`--selection-policy strict-boundary` when all primitives at the boundary
should be removed, which can retain fewer than the requested fraction.

Evaluate a trained or refined model on the held-out cameras:

```bash
python cli/evaluate.py \
  --scene-path outputs/bicycle/splataudit_20 \
  --source-path /path/to/mipnerf360/bicycle \
  --output outputs/bicycle/splataudit_20/eval \
  --save-images
```

Evaluation writes per-view and aggregate metrics to `results.json`, confidence
scores to `scores.npy`, and optional full-resolution renders and magma FAS
maps under the evaluation directory.

## Layout

- `cli/`: training, scoring, pruning/refinement, and evaluation
- `splataudit/geometry/`: support graph and the two SplatAudit signals
- `splataudit/backends/`: narrow adapters to the Graphdeco backend
- `splataudit/scene.py`: deterministic sparse-view scene filtering
