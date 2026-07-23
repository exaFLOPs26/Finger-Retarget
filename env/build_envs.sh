#!/bin/bash
# ============================================================================
# Do-As-I-Do — environment build recipe for a glibc-2.34 (Rocky 9.4) + Blackwell
# cluster. This is a REFERENCE of the exact steps that worked, not a turnkey
# script — read it, adapt paths, run section by section and check each verify.
#
# Prereqs: conda, system CUDA 12.8 at $CUDA_SYS (for compiling extensions),
# system gcc (glibc 2.34). Source env/daid_env.sh first (sets HOME->scratch,
# CONDA_ENVS_DIRS, caches, CC/CXX=system gcc).
# ============================================================================
set -eo pipefail
source "$(dirname "$0")/daid_env.sh"
REPO=/scratch/exaflops/do-as-i-do          # <-- upstream repo clone (adapt)
CUDA_SYS=/opt/ohpc/pub/apps/cuda/12.8      # <-- system CUDA 12.8 (adapt)
export CUDA_HOME=$CUDA_SYS PATH=$CUDA_SYS/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.9;12.0" # a100 ; 4090 ; pro6000 (add yours)
export FORCE_CUDA=1 MAX_JOBS=8
CU=https://download.pytorch.org/whl/cu128
pip_i(){ pip install --retries 20 --timeout 300 "$@"; }

# ---- GOTCHA: `conda create python=3.10` can resolve to GraalPy (no torch/wheels!)
#      Always force the CPython build string: python=3.X.*=*_cpython
CPY='=*_cpython'

# --------------------------- retargeting (py3.12) ---------------------------
conda create -y -n retargeting -c conda-forge "python=3.12.*$CPY"
conda activate retargeting
pip_i -e "$REPO/retargeting"          # MuJoCo Warp sampling-MPC; targets sharpa hand
python -c "import mujoco, warp; print('retargeting OK')"

# ------------------------------- sam3 (py3.12) ------------------------------
# Segmentation (SAM3). Its env/sam3.yml solves cleanly (py3.12 -> no GraalPy risk).
conda env create -n sam3 -f "$REPO/reconstruction/env/sam3.yml"
conda activate sam3 && conda install -y -c conda-forge ffmpeg

# --------------------------- daid_tapnet (py3.10) ---------------------------
conda create -y -n daid_tapnet -c conda-forge "python=3.10.*$CPY"
conda activate daid_tapnet && conda install -y -c conda-forge ffmpeg
pip_i torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url $CU
pip_i -e "$REPO/reconstruction/modules/tapnet[torch]"
pip install --only-binary=:all: einops tqdm mediapy matplotlib

# --------------------------- daid_hawor (py3.10) ----------------------------
conda create -y -n daid_hawor -c conda-forge "python=3.10.*$CPY"
conda activate daid_hawor && conda install -y -c conda-forge ffmpeg
pip_i torch==2.9.0 torchvision torchaudio --index-url $CU
# lietorch dispatch.h patch BEFORE compiling DROID-SLAM (see patches/):
git -C "$REPO/reconstruction/modules/HaWoR" apply /path/to/patches/lietorch_dispatch.h.diff || true
( cd "$REPO/reconstruction/modules/HaWoR/thirdparty/DROID-SLAM" && python setup.py install )
conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -n daid_hawor
# HaWoR requirements: install PER-PACKAGE skipping source-built ones (atomic
# `pip install -r` aborts on pytorch3d@git / torch-scatter). numpy pins to 1.26.4.
grep -vE "mmcv==1.3.9|chumpy@|^torch|pytorch3d|git\+" "$REPO/reconstruction/modules/HaWoR/requirements.txt" \
  | while read p; do [ -z "$p" ] && continue; pip install "$p" || echo "SKIP $p"; done
pip install "chumpy@git+https://github.com/mattloper/chumpy" --no-build-isolation
pip install "setuptools<81" && pip install pytorch-lightning==2.2.4 --no-deps && pip install lightning-utilities torchmetrics==1.4.0
# pytorch3d (build + glibc fix — see below); torch-scatter from the pyg wheel index:
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
# --> then build_pytorch3d daid_hawor (function at bottom)

# ------------------------------ sam3d (py3.11) ------------------------------
# GOTCHA: env/sam3d.yml pins sysroot_linux-64=2.39 (glibc 2.39) -> strip it:
grep -v "sysroot_linux-64=2.39" "$REPO/reconstruction/env/sam3d.yml" > /tmp/sam3d.glibc234.yml
conda env create -n sam3d -f /tmp/sam3d.glibc234.yml    # conda part; pip part may need the loop below
conda activate sam3d && conda install -y -c conda-forge ffmpeg
# The yml's pip manifest must be extracted CLEANLY (leading "- " breaks pip and it
# silently skips lines). Extract, then per-package install skipping source-built:
python - <<'PY' > /tmp/sam3d_pip.txt
import re
inpip=False
for l in open("REPO_reconstruction_env_sam3d.yml".replace("REPO_reconstruction_env","/scratch/exaflops/do-as-i-do/reconstruction/env/sam3d").replace("_yml",".yml")):  # adapt path
    if re.match(r'^\s*-\s*pip:\s*$', l): inpip=True; continue
    if inpip:
        m=re.match(r'^\s+-\s+(.+?)\s*$', l)
        if m: print(m.group(1))
        elif l.strip() and not l.startswith('  '): break
PY
export PIP_EXTRA_INDEX_URL=$CU
grep -vE "diff-gaussian-rasterization|pytorch3d" /tmp/sam3d_pip.txt \
  | while read p; do [ -z "$p" ] && continue; case "$p" in --*) continue;; esac; pip install "$p" || echo "SKIP $p"; done
# The special / non-PyPI ones:
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@5cb6a14~1"   # 0.0.2-era API (has utils3d.numpy.depth_edge)
pip install --no-deps "git+https://github.com/microsoft/MoGe.git@3c0d800"                 # moge 1.0.0 (MoGe-1)
pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
# diff-gaussian-rasterization (Mip-Splatting) — needed for SAM3D texture baking:
( git clone --recursive https://github.com/autonomousvision/mip-splatting.git /tmp/mips
  cd /tmp/mips/submodules/diff-gaussian-rasterization && pip install --no-build-isolation . )
pip install --no-deps -e "$REPO/reconstruction/modules/sam-3d-objects"   # --no-deps: else it clobbers torch to cu130
# sam3d_objects/__init__ imports a Meta-internal `init` module -> already handled by
# LIDRA_SKIP_INIT=true in config/paths.sh (see patches/paths.sh.diff).
# --> then build_pytorch3d sam3d

# ------------------- pytorch3d builder (per env; glibc fix) ------------------
build_pytorch3d(){  # $1 = env name
  conda activate "$1"
  pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall \
      "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
  pip install lief patchelf
  python /path/to/patches/pytorch3d_glibc_fix.py --auto      # rebind hypotf@GLIBC_2.35
  python -c "import torch; from pytorch3d import _C; from pytorch3d.renderer import look_at_view_transform; print('pytorch3d OK')"
}
# build_pytorch3d sam3d ; build_pytorch3d daid_hawor

echo "All envs built. Fetch weights (reconstruction/setup/02_fetch_weights.sh --download,"
echo "needs HF gated access to facebook/sam-3d-objects + facebook/sam3) and symlink MANO."
