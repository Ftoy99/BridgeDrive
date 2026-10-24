#!/usr/bin/env bash
# DD logit ladder on dd_k20_public_v2 (navtrain 20k fit -> navtest once). Plan: PhD/Selector_Results.md "DD logit ladder" (post-hoc).
# Launch: cd /mnt/jimmys/BridgeDrive && nohup bash test/selector/run_logit.sh > /mnt/ds/navsim-main/exp/selector/logit_driver.log 2>&1 &
# Each run dir is logged to MLflow: selector_train (selector) and selector_compare (paired bootstrap vs DD cls).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/harris/miniconda3/envs/hidden/bin/python
RUNS=/mnt/ds/navsim-main/exp/selector/runs
LOGGER="uv run --no-project --python 3.11 --with mlflow>=3.5 python test/selector/log_selector_run.py"

run() {  # run <phase_c args...>: train/eval, then log the run dir and its compare dir
  out=$($PY test/selector/phase_c.py "$@") || { echo "FAILED: $*"; return; }
  echo "$out"
  name=$(echo "$out" | awk '/^RUN_DIR/{print $2}')
  $LOGGER "$name" --experiment selector_train
  cmp=$RUNS/cmp_$(basename "$name")_vs_ddcls
  [ -d "$cmp" ] && $LOGGER "$cmp" --experiment selector_compare
}

for seed in 0 1 2; do
  run --selector a1l --loss pairwise --seed $seed
  run --selector r0l --loss pairwise --seed $seed
done
echo LOGIT_DONE
