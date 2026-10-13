#!/usr/bin/env bash
# Denoising visualization + timing + select/best/worst PDMS on N navtest scenes.
#   N_SCENES=4 bash test/run_viz_bridgedrive.sh      (CKPT=... GPU=0 VIZ_OUT=... to override)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVKIT="${REPO_ROOT}/BridgeDrive_adaptation_DiffusionDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/ds/navsim-main/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"
export PYTHONPATH="${DEVKIT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CKPT="${CKPT:-/mnt/ds/navsim-main/exp/training_bridgedrive_k80_beta10/2026.10.09.04.50.09/checkpoints/best-epoch045-valtraj3.797.ckpt}"
export BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
export ANCHOR_PATH="${ANCHOR_PATH:-${DEVKIT}/kmeans_navsim/navsim_anchor_k80.npy}"
cd "${DEVKIT}"
python "${REPO_ROOT}/test/viz_bridgedrive_denoise.py"
