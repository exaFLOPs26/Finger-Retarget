# do-as-i-do — cluster setup & reproduction notes

Setup scripts, patches, and reproduction notes for running **[Do-As-I-Do](https://github.com/malik-group/do-as-i-do)**
(Paliwal, Etukuru, Abbeel, Shafiullah, Malik) end-to-end on a **glibc-2.34 (Rocky 9.4) + NVIDIA Blackwell** cluster.

This repo does **not** contain the upstream code, model weights, or generated data — only the
glue needed to stand the pipeline up on hardware/OS it wasn't packaged for. Clone the upstream
repo separately and apply these.

> **Status:** the full pipeline was validated end-to-end here — raw video →
> reconstruction (SAM3 seg → SAM3D mesh → MoGe → HaWoR → GeoCalib → TAPIR → object
> tracking → layout) → retargeting (IK → MuJoCo-Warp physics opt) → Sharpa hand
> trajectory. Object tracking error ≈ 2.7 cm / 0.11 rad on the whisking demo.

## What's here
```
env/daid_env.sh          Environment redirect (see "Full /home" gotcha) + system-gcc export
env/build_envs.sh        Exact build recipe for all 5 conda envs (reference, adapt paths)
patches/paths.sh.diff            LIDRA_SKIP_INIT + input-dir note for reconstruction/config/paths.sh
patches/lietorch_dispatch.h.diff DROID-SLAM / lietorch fix for modern torch
patches/pytorch3d_glibc_fix.py   Rebind hypotf@GLIBC_2.35 -> 2.2.5 so pytorch3d loads on glibc 2.34
pipeline/run_pipeline_text.sh        Headless reconstruction driver (SAM3 text-seg, no GUI)
pipeline/run_pipeline_from_hawor.sh  Resume-from-HaWoR helper (skips finished Stage 0-2)
viz/render_autoframe.py  Render a retargeted trajectory to MP4 (OSMesa, auto-framed camera)
```

## Prerequisites
- conda; **system CUDA 12.8** (for compiling extensions); **system gcc** (glibc 2.34).
- GPU with ≥24 GB (a 4090 sufficed — no OOM, incl. the 17 GB SAM3D models; a bigger GPU only speeds the ~70-min diffusion tracking stage).
- **HF gated access** to `facebook/sam-3d-objects` and `facebook/sam3` (request on HuggingFace).
- **MANO** models (register at mano.is.tue.mpg.de — license forbids redistribution; you must obtain your own).

## The gotchas (why this repo exists)
1. **`conda create python=3.10` can resolve to GraalPy** (has no torch/numpy wheels → every
   `pip install torch` fails with "from versions: none"). Force CPython: `python=3.10.*=*_cpython`.
   (py3.12 envs are unaffected.)
2. **glibc 2.34.** (a) `env/sam3d.yml` pins `sysroot_linux-64=2.39` → the conda solve fails; strip
   that one line. (b) pytorch3d's `_C.so` requires `hypotf@GLIBC_2.35` (CUDA-12.8 build headers);
   the system libm only has it at 2.2.5 → rebind with `patches/pytorch3d_glibc_fix.py`. Build
   pytorch3d **per env** (sam3d py3.11/torch2.8, daid_hawor py3.10/torch2.9).
3. **sam3d pip manifest.** The yml's pip block, if naively extracted, keeps a `- ` prefix that pip
   silently skips → dozens of missing deps. Extract cleanly, then **per-package** install skipping
   source-built ones (an atomic `pip install -r` aborts on the first unbuildable pin).
   Exact non-PyPI pins: **utils3d** git `5cb6a14~1` (newer versions drop `utils3d.numpy.depth_edge`),
   **moge** git `3c0d800` (=1.0.0 / MoGe-1), **kaolin** 0.18.0 (NVIDIA index), **nvdiffrast** (git,
   `--no-build-isolation`), **diff-gaussian-rasterization** (Mip-Splatting — needed for texture baking).
   `pip install -e sam-3d-objects --no-deps` (else it clobbers torch to cu130).
4. **daid_hawor.** HaWoR `requirements.txt` install aborts atomically on `pytorch3d@git` /
   `torch-scatter`; exclude + loop. numpy pins to **1.26.4** (chumpy); droid_backends still imports.
   torch-scatter from the pyg wheel `data.pyg.org/whl/torch-2.9.0+cu128.html`.
5. **`LIDRA_SKIP_INIT=true`** — `sam3d_objects/__init__` imports a Meta-internal module absent from
   the fork; the flag skips it (baked into `config/paths.sh`, see the patch).
6. **Headless SAM3.** Upstream Stage-1 object seg uses a click GUI; `run_pipeline_text.sh` swaps it
   for `run_sam3_video.py --text "<object>"` (SAM3 text prompt) so it runs with no display.
7. **Input dir naming.** The reconstruction infers paths from the video's *parent-dir basename*, so
   the dir must be named after the video stem: put `foo.mp4` in `foo/`, not `anything_else/`.
8. **Full `/home`.** All build/runtime home-writes are redirected to `/scratch` via `daid_env.sh`
   (conda/pip/HF/XDG caches). Adapt the paths to yours.

## Usage
```bash
# 0) clone upstream (with submodules) + apply patches
git clone --recursive https://github.com/malik-group/do-as-i-do.git
cd do-as-i-do
git apply /path/to/patches/paths.sh.diff
git -C reconstruction/modules/HaWoR apply /path/to/patches/lietorch_dispatch.h.diff

# 1) build the 5 envs  (edit paths first)
bash /path/to/env/build_envs.sh

# 2) fetch weights (needs HF gated access) + symlink your MANO
cd reconstruction && ./setup/02_fetch_weights.sh --download

# 3) reconstruct a clip  (dir MUST be named after the video stem)
source /path/to/env/daid_env.sh
./run_pipeline_text.sh mytask/mytask.mp4 <ref_frame> <object> <right|left>

# 4) retarget -> Sharpa trajectory
cd ../retargeting
python launch.py --task mytask --raw-dir ../reconstruction/mytask --no-show-viewer

# 5) visualize
python replay_viser.py --run-dir outputs/sharpa/right/mytask/0          # interactive 3D (port-forward)
python /path/to/viz/render_autoframe.py                                  # or an auto-framed MP4 (edit run dir)
```

## Notes on quality (for downstream work)
The retargeting sampling-MPC optimizes primarily for **object tracking** (reward = object pos/quat
error). **Finger motion** just follows the MANO→Sharpa kinematic map with contact approximations, so
fine dexterous finger detail is the weakest part of the output — a natural target for a residual /
contact-aware policy layered on top of these open-loop trajectories.

## Attribution & licenses
Upstream code: [malik-group/do-as-i-do](https://github.com/malik-group/do-as-i-do) (and its
submodules) — see their LICENSE; this repo only adds setup glue and does not relicense their work.
Weights (SAM3/SAM3D, HaWoR) are gated/licensed by their owners; **MANO** redistribution is
prohibited — everyone must obtain their own. Nothing licensed/gated is included here.
