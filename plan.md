# BridgeDrive K80 (paper config) — plan & findings

## PDM eval of training checkpoints (`test/run_pdm_fast_bridgedrive.sh`)

| epoch | SELECT (deployed) | BEST (oracle) | gap | WORST | pool mean |
|---|---|---|---|---|---|
| 10  | 0.8194 | 0.9645 | 0.145 | 0.0242 | 0.6826 |
| 30  | 0.8593 | 0.9486 | 0.089 | 0.1850 | n/a (not logged) |
| 70  | 0.8678 | 0.9419 | 0.074 | 0.3155 | 0.7850 |
| 150 | 0.8712 | 0.9451 | 0.074 | 0.2232 | 0.7720 |

Pool size 80 (K80). SELECT = classifier pick; BEST = oracle max-PDMS in pool.

### SELECT subscores (ep150)
NC 0.9815, DAC 0.9555, EP 0.8164, TTC 0.9394, Comfort 0.9987, DDC 0.9809.
Oracle (ep150): NC 0.9938, DAC 0.9943, EP 0.8988, TTC 0.9790.

## Findings
- Deployed PDMS plateaus: +4.0 (ep10->30), +0.85 (30->70), +0.34 (70->150). Little gain past ~70 epochs.
- Selector is bottleneck: oracle ceiling flat (~0.945), gap to deployed stuck at 0.074 since ep70.
- Pool gets tighter: worst-case rises (0.02 -> 0.32 @ep70), oracle slightly falls. Generator sharpens, less diversity.
- Biggest losses vs oracle: ego_progress (0.816 vs 0.899, selector too conservative), TTC (0.939 vs 0.979), DAC (0.956 vs 0.994).

## Next steps
1. Improve selector, not generator: bigger scorer head, PDMS regression and/or ranking loss over the pool.
2. Check classifier labels = per-candidate PDMS; upweight EP and DAC terms.
3. Stop training at ~70-100 epochs to save compute.
4. Log pool mean for ep30 (missing).
