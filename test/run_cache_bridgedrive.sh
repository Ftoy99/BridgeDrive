#!/usr/bin/env bash
# Cache NAVSIM navtrain features/targets for BridgeDrive (k80, beta10 = paper config).
# Uses the agent's own feature/target builders (agent=bridgedrive_agent_k80_beta10),
# writes to a NEW cache dir and never touches existing exp/ folders.
#
# Usage:
#   bash test/run_cache_bridgedrive.sh                 # full navtrain cache
#   SMOKE=20 bash test/run_cache_bridgedrive.sh        # 20-scene smoke test -> separate dir
# Re-running resumes: tokens already cached are skipped (force_cache_computation=False).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVKIT="${REPO_ROOT}/BridgeDrive_adaptation_DiffusionDrive"

# ---- data locations (override via env) ----
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/ds/navsim-main/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/ds/navsim-main/exp}"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"          # this repo's navsim, not DiffusionDriveV2
export PYTHONPATH="${DEVKIT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# ---- local weights / anchors ----
BKB_PATH="${BKB_PATH:-${REPO_ROOT}/weights/resnet34.a1_in1k/pytorch_model.bin}"
ANCHOR_PATH="${ANCHOR_PATH:-${DEVKIT}/kmeans_navsim/navsim_anchor_k80.npy}"

# ---- output (new dir; never an existing user cache) ----
CACHE_NAME="${CACHE_NAME:-training_cache_bridgedrive_k80}"
EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
  CACHE_NAME="${CACHE_NAME}_smoke${SMOKE}"
  EXTRA+=("train_test_split.scene_filter.max_scenes=${SMOKE}")
fi
CACHE_PATH="${NAVSIM_EXP_ROOT}/${CACHE_NAME}"

# guard: refuse names of pre-existing caches that hold other formats
case "${CACHE_NAME}" in
  training_cache|training_cache_temporal|cache_hidden_temporal|metric_cache|metric_cache_navtrain|pretrained)
    echo "Refusing to use existing cache dir name: ${CACHE_NAME}" >&2; exit 1;;
esac

for p in "${BKB_PATH}" "${ANCHOR_PATH}" "${NUPLAN_MAPS_ROOT}" "${OPENSCENE_DATA_ROOT}/navsim_logs/trainval" "${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval"; do
  [[ -e "$p" ]] || { echo "Missing: $p" >&2; exit 1; }
done

mkdir -p "${CACHE_PATH}"
echo "devkit : ${DEVKIT}"
echo "cache  : ${CACHE_PATH}"
echo "threads: ${THREADS:-all}"

cd "${DEVKIT}"
python navsim/planning/script/run_dataset_caching.py \
  agent=bridgedrive_agent_k80_beta10 \
  +agent.config.bkb_path="${BKB_PATH}" \
  agent.config.plan_anchor_path="${ANCHOR_PATH}" \
  experiment_name=cache_bridgedrive_k80 \
  train_test_split=navtrain \
  split=trainval \
  cache_path="${CACHE_PATH}" \
  force_cache_computation=False \
  ${THREADS:+worker.threads_per_node=${THREADS}} \
  "${EXTRA[@]}"

echo "Done. Cache: ${CACHE_PATH}  ($(find "${CACHE_PATH}" -name '*.gz' | wc -l) files)"
