#!/usr/bin/env bash
# §5f S2 feature cache: frozen public DiffusionDrive features (camera, LiDAR, status) + targets, navtest, current frame.
# Uses the public devkit's own caching script (run_dataset_caching.py) so features match the DD agent exactly.
# New cache dir only. Re-running resumes (force_cache_computation=False).
#   SMOKE=20 bash test/selector/cache_dd_sensor.sh     # 20-scene smoke -> separate dir
#   bash test/selector/cache_dd_sensor.sh               # full navtest (12,146 scenes)
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
CKPT_BKB="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-/home/harris/Desktop/Jimmys/DiffusionDriveV2/kmeans_navsim_traj_20.npy}"
CACHE_NAME="${CACHE_NAME:-sensor_cache_dd_navtest}"
EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
  CACHE_NAME="${CACHE_NAME}_smoke${SMOKE}"
  EXTRA+=("train_test_split.scene_filter.max_scenes=${SMOKE}")
fi
case "${CACHE_NAME}" in
  training_cache|training_cache_temporal|cache_hidden_temporal|metric_cache|metric_cache_navtrain|pretrained|training_cache_bridgedrive_k80)
    echo "Refusing existing cache dir name: ${CACHE_NAME}" >&2; exit 1;;
esac
CACHE_PATH="${NAVSIM_EXP_ROOT}/${CACHE_NAME}"
mkdir -p "${CACHE_PATH}"
echo "cache: ${CACHE_PATH}"
cd "${REPO_ROOT}"
"${PY}" "${DEVKIT}/navsim/planning/script/run_dataset_caching.py" \
  agent=diffusiondrive_agent \
  +agent.config.bkb_path="${CKPT_BKB}" \
  +agent.config.plan_anchor_path="${ANCHOR_PATH}" \
  experiment_name=sensor_cache_dd_navtest \
  train_test_split=navtest \
  split=test \
  cache_path="${CACHE_PATH}" \
  force_cache_computation=False \
  ${THREADS:+worker.threads_per_node=${THREADS}} \
  "${EXTRA[@]}"
echo "Done: $(find "${CACHE_PATH}" -name '*.gz' | wc -l) feature files"
