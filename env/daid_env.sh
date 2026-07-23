#!/bin/bash
# Do-As-I-Do build/run environment. /home is full (200G, 0 free), so redirect ALL
# home-writes (conda config/metadata, pip cache, HF cache, XDG cache) to /scratch.
# Source this before any conda build OR pipeline run for do-as-i-do.
export HOME=/scratch/exaflops/daid_home
export CONDARC=$HOME/.condarc
export CONDA_PKGS_DIRS=/scratch/exaflops/conda_pkgs
export CONDA_ENVS_DIRS=/scratch/exaflops/conda_envs
export PIP_CACHE_DIR=/scratch/exaflops/pip_cache
export HF_HOME=/scratch/exaflops/hf_cache
export XDG_CACHE_HOME=$HOME/.cache
mkdir -p "$HOME" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
source /home/exaflops/miniconda3/etc/profile.d/conda.sh

# Force system compiler (glibc 2.34) so built/JIT CUDA exts do not require GLIBC_2.35
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
