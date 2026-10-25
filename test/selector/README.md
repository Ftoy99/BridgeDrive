# Selector tests (NAVSIM trajectory selection)

Code for the selector study on the public DiffusionDrive pool (`dd_k20_public_v2`, navtest, K=20 candidates per scene). Plans, pre-registrations and every result with its MLflow run id live in the vault: `PhD/Selector_Experiment_Plan.md` (plan of record), `PhD/Selector_Results.md` (results), `PhD/Selector_Runbook.md` (run naming, sanity gates), `PhD/Selector_Tables.md` (locked tables). Large outputs live on the work PC under `/mnt/ds/navsim-main/exp/selector/` and are not in git.

## Rules the code follows

- **Public checkpoints only** for baselines and pools (DiffusionDrive `diffusiondrive_navsim_88p1_PDMS`).
- **Privileged inputs** (GT future, GT agents, map DAC, GT BEV) are for teachers, labels or diagnostics only. Tag `privileged=1`. Never a deployable selector input.
- **Navtest is read once per final config.** Learned heads choose their threshold or epoch on navtrain (or a calibration split by log), then score navtest once.
- **Paired bootstrap** (20k resamples) by scene, plus cluster bootstrap by `log_name`, for every Δ vs DD cls.
- **Negative results stay.** Every run gets a row in `Selector_Results.md`.

## Pipeline (in order)

| Step | Script | Launcher | Output |
|---|---|---|---|
| 1. DD pool dump: K=20 candidates, DD logit, official PDMS sub-scores | `dump_dd_pool.py` | `dump_dd_pool.sh` | `dd_k20_public_v2/{navtest,navtrain}/shard_*.npz` |
| 2. Map DAC gate (CPU, train-free) | `map_gate.py`, `map_gate_score.py` | — | per-candidate DAC |
| 3. Sensor cache (camera + LiDAR, current frame and lags) | `cache_dd_history.py` | `cache_dd_history.sh`, `cache_dd_sensor.sh` | `sensor_cache_dd_*` |
| 4. Pool sensor features (BEV pooled 8×8, lags 0–3) | `pool_sensor_feats.py` | `pool_sensor_feats.sh` | `sensor_pool/sensor_pool.npz` |
| 5. Scorer-simulated poses (LQR, V3 input) | `sim_poses.py` | — | `diag_bev_vs_map/sim_*` |
| 6. GT semantic BEV (labels only) | `gt_bev.py` | — | `diag_bev_vs_map/gtbev_*` |

### Experiments

| Experiment | Script | Launcher | Notes |
|---|---|---|---|
| Phase C selector zoo (learned heads a0/a1/r3, filter-then-rank, DD cls) | `phase_c.py` | `run_phase_c.sh` | navtrain 90/10 calibration, navtest once |
| Logit ladder (DD logit as input) | — | `run_logit.sh` | post-hoc, `post_hoc=1` |
| Privileged ladder | — | `run_privileged.sh` | `privileged=1` diagnostics |
| §5d cross-fit by log, 3 seeds | `crossfit_5d.py` | `run_crossfit_5d.sh` | pre-registered in plan §5d |
| §5f / §5g sensor ladder (S1–S3, tuned) | `sensor_rungs.py`, `sensor_rungs_helpers.py`, `sensor_rungs_train.py` | — | cross-fit by log |
| §5h data-size test, navtrain-20k | `sensor_rungs_train.py` | `run_datasize_5h.sh` | one navtest read per rung |
| §5i BEV vs map check, gate variants V0–V3 | `bev_vs_map.py` | — | run dir `runs/sel_dd_k20_public_v2_5i_bevmap_<V>_…` |
| §5j footprint-sampled DD BEV probabilities (Amendment A1) | `footprint_feats.py`, `footprint_j1.py` | `footprint_feats.sh` | features on navtrain-20k then navtest; J1 head; run dir `runs/…_5j_J1_…` |
| Information probe, gate ceilings, headroom audit | `info_probe.py`, `gate_ceiling.py`, `audit_headroom.py` | — | privileged diagnostics and F1 headroom |
| A4.3 forgiveness check, A5 train/test shift | `forgive_check.py`, `a5_shift.py` | — | diagnostics |
| Soft tilt on observable terms (§5e) | `tilt_select.py`, `observable_terms.py` | — | pre-registered §5e |
| Pool structure audit | `audit_pool_structure.py` | — | structure only |
| GTRS vocabulary labels (P2/P3) | `convert_gtrs_labels.py`, `make_manifest.py` | — | 8192 / 16384 vocabularies |

Smoke or partial runs carry `smoke=1` in their tags.

## Logging to MLflow

Every run writes a run dir with `params.json` (flat), `metrics.json` (flat, numeric), `tags.json`, and `result.json` if needed. Then log it:

```bash
uv run --no-project --python 3.11 --with 'mlflow>=3.5' python test/selector/log_selector_run.py RUN_DIR --experiment <name>
```

Store: `sqlite:////mnt/ds/mlflow/mlflow.db`, UI on `http://127.0.0.1:5000`. Experiment names in use: `selector_audit`, `selector_diag`, `selector_sensor_ladder`, `selector_datasize`. A new kind of run gets its own experiment name before it is launched.

Feature builds (pipeline steps 3–6) are data steps. Their metadata lives in their driver logs, not in MLflow.

## Environment

- Python: `/home/harris/miniconda3/envs/hidden/bin/python` (the `hidden` env). DiffusionDrive devkit at `/mnt/jimmys/public_repos/DiffusionDrive`.
- Env vars set by the launchers: `NUPLAN_MAP_VERSION=nuplan-maps-v1.0`, `OPENSCENE_DATA_ROOT=/mnt/ds/navsim-main/dataset`, `NAVSIM_EXP_ROOT=/mnt/ds/navsim-main/exp`, `NAVSIM_DEVKIT_ROOT`, `PYTHONPATH`, `CUDA_VISIBLE_DEVICES`.
- Backbone weights: `weights/resnet34.a1_in1k/` (ignored by git, public).
- Before any GPU run: `ps aux | grep -E 'cache_dd_history|pool_sensor|sensor_rungs|crossfit|phase_c|footprint'` and read the live log. Do not start a GPU job while another row in the vault's Active runs is running.

## Launch example

```bash
cd /mnt/jimmys/BridgeDrive
SPLIT=navtrain20k bash test/selector/footprint_feats.sh > $NAVSIM_EXP_ROOT/selector/footprint_feats/navtrain20k_driver.log 2>&1
SPLIT=navtest     bash test/selector/footprint_feats.sh
/home/harris/miniconda3/envs/hidden/bin/python test/selector/footprint_j1.py
```

Record the exact launch command and env in the vault's area log (`PhD/Activity_Log.md`) at launch.

## Known caveats

- DiffusionDrive was trained on navtrain. Heads trained on navtrain see in-sample DD outputs. Negatives are not affected. Positive results would need a check.
- Gate geometry: the devkit box (5.176 × 2.297 m, centre +1.461 m) is used in §5i and §5j. The approximate box in `phase_c.py` (4.9 × 2.0 m) is kept for comparison only.
- `cache_dd_history.sh` builds `CACHE_ROOT` from `NAME`, so a `CACHE_ROOT` env var is ignored there. Use `NAME=`.
- `footprint_j1.py` used a fallback threshold (max precision) for its one navtest read, because no threshold reached the 0.95 precision gate on the navtrain inner split. This is recorded in its `result.json`.
