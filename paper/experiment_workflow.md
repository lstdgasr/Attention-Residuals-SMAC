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

## Run Follow-Up Experiments

Use this queue to add `3s5z` cross-algorithm validation, seed4/5 for the core QMIX comparisons, and attention-weight logging for `qmix_attnres_l2`:

```bash
cd /home/xhl009/MARL/pymarl
conda activate pymarl-sc2
export SC2PATH=/home/xhl009/MARL/pymarl/3rdparty/StarCraftII_srv
export RESULTS_DIR=/home/xhl009/MARL/pymarl-results

DRY_RUN=1 SESSION_PREFIX=followup6 bash scripts/run_marl_transfer_followup_6gpu.sh
RUN_SMOKE_TEST=0 SESSION_PREFIX=followup6 bash scripts/run_marl_transfer_followup_6gpu.sh
```

## Monitor

```bash
tmux ls
watch -n 5 nvidia-smi
tail -f "$RESULTS_DIR/launcher_logs/3s5z_iql_attnres_l2_s3_gpu0.log"
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

Generate paper figures:

```bash
python scripts/plot_marl_transfer_curves.py \
  --sacred-dir results/sacred \
  --output-dir paper/latex/figures

python scripts/plot_attn_weight_heatmaps.py \
  --sacred-dir results/sacred \
  --output-dir paper/latex/figures \
  --configs qmix_attnres_l2
```

## Current Missing or Partial Slots

From the latest local sync:

```text
No missing 3-seed slots in the completed 5m_vs_6m adaptation matrix.
```

The VDN baseline is complete and can be used in the `5m_vs_6m` cross-algorithm paired comparison. The follow-up queue intentionally adds expected seed4/5 slots and `3s5z` IQL/VDN cross-algorithm slots; these will appear in `marl_transfer_missing_or_partial.csv` until `scripts/run_marl_transfer_followup_6gpu.sh` finishes and summaries are regenerated.

## Interpretation Rule

Use QMIX as the primary complete ablation. Use IQL/VDN only as cross-algorithm validation for the lightweight `AttnRes-L2` transfer. Do not claim that AttnRes is a strong SMAC algorithm unless final win, AUC, and seed-paired comparisons are consistently positive.
