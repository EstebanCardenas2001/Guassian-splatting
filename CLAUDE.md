# Gaussian Splatting of a potted plant, built from scratch on a T4

Physics of Data master's project: build 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)
progressively and mostly from scratch, ending with the highest-fidelity reconstruction a single
NVIDIA T4 can produce of a golden pothos in an unglazed terracotta pot.

Two equal goals: **understanding** (every stage is my own code, tested and explained before moving on)
and **final quality** (sharp leaf edges, thin stems, terracotta texture, view-dependent leaf sheen).

## Environment (as set up in Stage 0)

- CloudVeneto VM, headless Ubuntu 26.04. Visuals go to files or SSH port forwards (TensorBoard 6006, Jupyter, viser).
- Tesla T4: 15 GB, sm_75, ~8 TFLOPS FP32, FP16 tensor cores, **no bf16**. Driver 580.178.04 (CUDA ≤ 13.0).
- 15 CPU cores, 88 GiB RAM, no swap.
- **Disk:** root `/` has only ~2.6 GB free, so don't put anything large there. Everything heavy lives on the 500 GB
  ext4 volume `/mnt`: `/mnt/miniforge3`, `/mnt/cache/{pip,conda-pkgs,torch}`, `/mnt/data`, `/mnt/outputs`.
  `data/` and `outputs/` in the repo are symlinks to those.
- Conda env `gsplat`: Python 3.12, torch 2.14.1+cu126, nvcc 12.6, gcc 13 (system gcc 15 is too new for nvcc 12.6).
  The activate script sets `TORCH_CUDA_ARCH_LIST=7.5`, `CUDA_HOME`, `CC`/`CXX`.
- COLMAP from apt (3.12.6, CPU-only build: use `--SiftExtraction.use_gpu 0 --SiftMatching.use_gpu 0`).
- Setup: `scripts/setup_env.sh`; sanity check: `python scripts/check_env.py`; TensorBoard: `scripts/tensorboard.sh`.
- The VM can be interrupted: long jobs run in `tmux` and checkpoint regularly.

## Data

- ~150–200 Nikon D7200 photos (manual exposure, fixed WB/focus, f/8–f/11, ~35 mm) → `data/plant/raw/`.
- Validation first on NeRF-Synthetic Blender (`lego`) → `data/nerf_synthetic/`.
- Plant poses: COLMAP headless, single shared OPENCV camera, exhaustive matching.
- Foreground masks (plant + pot): rembg first, SAM if needed.

## How to work

- **One stage at a time.** Don't start the next stage until the current one meets its acceptance criteria
  and I've confirmed. End each stage with a short summary: what was built, key equations, results, surprises.
- **Explain the physics/math while implementing it.** Docstrings state shapes, coordinate conventions
  (world/camera/NDC/pixel), and which side of each matrix multiplication.
- **Tests first for the renderer:** `pytest` for projection, covariance, compositing; `torch.autograd.gradcheck`
  on tiny float64 cases.
- **Reproducibility:** one YAML per experiment in `configs/` (inherit with `_base_: base.yaml`), fixed seeds,
  git commit logged per run. Each run writes `outputs/<date>_<name>/` with config, metrics JSON,
  TensorBoard logs, validation renders (`src/run.py: make_run_dir`).
- **Commit after each working step** with a clear message.
- **Ask before** large downloads, installing system packages, or deleting data/outputs.
- **Don't fake results.** If something is slow, wrong or OOM, say so and show profiler/memory numbers.

## Repo layout

```
configs/            one YAML per experiment
data/ -> /mnt/data  raw photos, downscaled images, masks, COLMAP sparse/, nerf_synthetic/
outputs/ -> /mnt/outputs
scripts/            setup_env.sh, check_env.py, tensorboard.sh
src/
  config.py         YAML loading (+ _base_ inheritance, dotlist overrides)
  run.py            seeding, git provenance, run directories, metrics.json
  camera.py         intrinsics/extrinsics, COLMAP + Blender loaders
  gaussians.py      parameter container, activations, init (SfM / random box)
  render_torch.py   pure-PyTorch reference renderer
  render_gsplat.py  wrapper around gsplat's rasterizer (Stage 5)
  losses.py         L1, D-SSIM, mask/opacity losses
  densify.py        clone/split/prune, opacity reset, MCMC variant
  train.py
  evaluate.py       PSNR / SSIM / LPIPS, render test views
  export.py         .ply in standard 3DGS layout, fly-through video
tests/
notebooks/
```

## Stages and acceptance criteria

**Stage 0: Environment and scaffolding.** Report hardware; conda env, repo skeleton, config system; COLMAP installed.
Done when `pytest` runs, CUDA is visible from PyTorch, TensorBoard is reachable through a port forward.

**Stage 1: 2D Gaussian image fitting.** Fit one plant photo with N 2D Gaussians (mean, rotation angle, log-scales,
RGB, sigmoid opacity). Additive blending first, then sorted alpha compositing. Sweep N ∈ {1k, 5k, 20k}, plot PSNR vs N.
Done when PSNR > 30 dB at 20k Gaussians on a 512 px crop.

**Stage 2: 3D forward model, pure PyTorch.** Pinhole projection; Σ = R S Sᵀ Rᵀ from normalized quaternion and
log-scales; EWA Σ' = J W Σ Wᵀ Jᵀ + 0.3 px² I; frustum and near-plane culling, depth sort, front-to-back
compositing C = Σ cᵢ αᵢ ∏_{j<i}(1−αⱼ) with early termination.
Done when unit tests and `gradcheck` pass and hand-placed Gaussians render as expected from several views.

**Stage 3: Synthetic inverse problem.** Render ~50 views of 200 known Gaussians, recover them from a random start.
Done when parameters converge and any failures (scale/rotation ambiguities) are explained.

**Stage 4: Blender lego with my renderer.** `transforms.json`, white background, L1 + 0.2·D-SSIM at a reduced
resolution, per-parameter LRs with exponential decay on position LR.
Done when lego is recognizable and test PSNR is recorded. Profile memory and time/iter; report the Gaussian
count and resolution at which the pure-PyTorch renderer breaks on the T4.

**Stage 5: Tiled CUDA rasterizer (gsplat), keep my training loop.** gsplat compiled for sm_75;
`render_gsplat.py` with the same interface as `render_torch.py`; numerical agreement on small scenes.
Optional: own tile-binning + compositing kernel (Triton or CUDA extension), benchmarked vs gsplat.
Done when lego reaches ≳ 32 dB PSNR at 800×800.

**Stage 6: Plant data pipeline.** Downscale to 1600 and 2000 px long side; COLMAP headless; `model_analyzer`
(≥ 95% registered, reprojection error); view sparse cloud in viser; masks spot-checked; loader with every 8th
image held out. Done when cameras/points look right in viser and masks align.

**Stage 7: Plant training, one feature at a time (each a saved run).**
A baseline (SfM init, RGB, fixed bg) · B + mask/opacity loss outside plant · C + adaptive density control
(view-space grad threshold, clone/split/prune, opacity reset) · D + SH degree 0→3 progressively ·
E random box init instead of SfM.

**Stage 8: Push the T4 to maximum detail.** Densification: standard vs AbsGS-style vs 3DGS-MCMC with a hard cap.
Anti-aliasing: gsplat antialiased mode, check leaf edges aren't blurred. Memory: packed mode, sparse grads,
images cached on GPU as uint8 if they fit (else pinned CPU); params in FP32, no bf16. Find the max training
resolution (1600 → 2000 → full) and Gaussian count with headroom in 16 GB. Report it/s, peak VRAM
(`torch.cuda.max_memory_allocated`), total time. 30k iterations baseline; test longer schedules.
Done when there is a table of resolution × cap × densification vs PSNR/SSIM/LPIPS/VRAM/time; best run chosen
on held-out LPIPS.

**Stage 9: Evaluation and deliverables.** Ablation table (A–E + best Stage 8); side-by-side crops of leaf edges
and stems; `.ply` in standard 3DGS layout (opens in SuperSplat); smooth orbit MP4 from interpolated poses;
README with method, equations, results, lessons for a room-scale capture.
