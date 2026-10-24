#!/usr/bin/env bash
# §5f S3 history cache launcher (lags 1-3 of navtest, DD camera+LiDAR features). Writes only new dirs.
#   SMOKE=20 bash test/selector/cache_dd_history.sh   # smoke -> separate dir
#   bash test/selector/cache_dd_history.sh             # full navtest
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVKIT="/mnt/jimmys/public_repos/DiffusionDrive"
PY="${PY:-/home/harris/miniconda3/envs/hidden/bin/python}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/ds/navsim-main/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"
export PYTHONPATH="${DEVKIT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export N_THREADS="${N_THREADS:-20}"
ANCHOR_PATH="${ANCHOR_PATH:-/home/harris/Desktop/Jimmys/DiffusionDriveV2/kmeans_navsim_traj_20.npy}"
BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
NAME="${NAME:-sensor_cache_dd_navtest_hist}"
EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
  NAME="${NAME}_smoke${SMOKE}"
  EXTRA+=("train_test_split.scene_filter.max_scenes=${SMOKE}")
fi
export CACHE_ROOT="${NAVSIM_EXP_ROOT}/${NAME}"
mkdir -p "${CACHE_ROOT}"
echo "cache root: ${CACHE_ROOT}"
cd "${REPO_ROOT}"
"${PY}" test/selector/cache_dd_history.py \
  train_test_split=navtest agent=diffusiondrive_agent \
  +agent.config.bkb_path="${BKB_PATH}" +agent.config.plan_anchor_path="${ANCHOR_PATH}" \
  experiment_name=sensor_cache_dd_history "${EXTRA[@]}"
