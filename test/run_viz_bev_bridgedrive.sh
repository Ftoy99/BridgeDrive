#!/usr/bin/env bash
# BEV overlay of saved candidates.
#   bash test/run_viz_bev_bridgedrive.sh   (after run_viz_bridgedrive.sh; reads test/viz_out/*.npz)
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
cd "${DEVKIT}"
python "${REPO_ROOT}/test/viz_bev_bridgedrive.py"
