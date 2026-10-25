#!/usr/bin/env bash
# §5d cross-fit on navtest (Selector_Experiment_Plan §5d). 3 seeds x {a1, a1l, hgb_a1l}, 2-fold by log_name.
# Launch: cd /mnt/jimmys/BridgeDrive && nohup bash test/selector/run_crossfit_5d.sh > /mnt/ds/navsim-main/exp/selector/crossfit_5d_driver.log 2>&1 &
# Each run dir -> MLflow selector_crossfit as it finishes; its cmp_ dir -> selector_compare.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/harris/miniconda3/envs/hidden/bin/python
LOGGER="uv run --no-project --python 3.11 --with mlflow>=3.5 python test/selector/log_selector_run.py"

$PY -u test/selector/crossfit_5d.py --seeds 0,1,2 | while IFS= read -r line; do
  echo "$line"
  case "$line" in
    RUN_DIR\ *)
      name=${line#RUN_DIR }
      $LOGGER "$name" --experiment selector_crossfit
      cmp=/mnt/ds/navsim-main/exp/selector/runs/cmp_$(basename "$name")_vs_ddcls
      [ -d "$cmp" ] && $LOGGER "$cmp" --experiment selector_compare
      ;;
  esac
done
echo CROSSFIT_5D_LOGGED
