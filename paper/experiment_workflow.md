# Experiment Workflow for the MARL Transfer Adaptation Study

## Server Setup

```bash
cd /home/xhl009/MARL/pymarl
conda activate pymarl-sc2
export SC2PATH=/home/xhl009/MARL/pymarl/3rdparty/StarCraftII_srv
export RESULTS_DIR=/home/xhl009/MARL/pymarl-results
mkdir -p "$RESULTS_DIR/launcher_logs"
```

## Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE=0 bash scripts/server_smoke_test.sh
```

## Run Remaining Adaptation Experiments

Preview the 18 planned jobs:

```bash
DRY_RUN=1 bash scripts/run_marl_transfer_adaptation_7gpu.sh
```

Launch jobs on GPU `0,1,3,4,5,6,7`:

```bash
bash scripts/run_marl_transfer_adaptation_7gpu.sh
```

Skip smoke test if it already passed:

```bash
RUN_SMOKE_TEST=0 bash scripts/run_marl_transfer_adaptation_7gpu.sh
```

## Monitor

```bash
tmux ls
watch -n 5 nvidia-smi
tail -f "$RESULTS_DIR/launcher_logs/5m_vs_6m_iql_attnres_l2_s1_gpu4.log"
```

## Summarize

```bash
python scripts/summarize_marl_transfer_adaptation.py \
  --sacred-dir results/sacred \
  --output-dir "$RESULTS_DIR/diagnostics"
```

Generated files:

```text
marl_transfer_primary_qmix_table.csv
marl_transfer_cross_algorithm_pairs.csv
marl_transfer_cross_algorithm_aggregate.csv
marl_transfer_missing_or_partial.csv
```

## Current Missing or Partial Slots

From the latest local sync:

```text
5m_vs_6m vdn seed1: missing
5m_vs_6m vdn seed2: missing
5m_vs_6m vdn seed3: missing
```

The QMIX main line is complete for `5m_vs_6m` and `3s5z`. The IQL and QMIX cross-algorithm lightweight AttnRes-L2 comparisons are complete. VDN cannot be used as a paired comparison until the missing baseline runs are added.

## Interpretation Rule

Use QMIX as the primary complete ablation. Use IQL/VDN only as cross-algorithm validation for the lightweight `AttnRes-L2` transfer. Do not claim that AttnRes is a strong SMAC algorithm unless final win, AUC, and seed-paired comparisons are consistently positive.
