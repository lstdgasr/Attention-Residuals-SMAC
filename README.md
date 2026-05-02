# Attention Residuals for SMAC Multi-Agent Reinforcement Learning

This repository is a PyMARL-based research codebase for studying whether
**Attention Residuals**, originally proposed for improving information flow in
large language models, can be transferred to shallow recurrent agents in
multi-agent reinforcement learning (MARL).

The project focuses on the StarCraft Multi-Agent Challenge (SMAC), especially
`5m_vs_6m` and `3s5z`. The goal is not to claim a new state-of-the-art SMAC
algorithm. Instead, this code supports an empirical adaptation study:

> Heavy LLM-style depth residual modules do not transfer reliably to shallow
> recurrent MARL agents; lightweight variants show limited learning signals, but
> direct transfer remains unstable and costly.

## What This Repository Adds

This project extends the original PyMARL codebase with agent-side variants for
Attention Residuals:

- **AttnRes-RNN agent**: inserts a depth-wise Attention Residual module between
  the GRU hidden state and the Q head.
- **AttnRes-L2**: a lightweight two-layer variant for lower-cost adaptation.
- **Block AttnRes**: a block-wise variant inspired by block residual aggregation.
- **Depth-MLP control**: adds comparable post-GRU MLP depth without depth
  attention, used to test whether gains/losses come from attention or from
  extra network depth.
- **Cross-algorithm checks**: lightweight AttnRes variants for QMIX, IQL, and
  VDN configs. Current paired analysis includes IQL and QMIX; VDN baseline runs
  are not complete, so VDN is not used as a paired conclusion.

## Repository Layout

```text
src/modules/agents/
  attn_residual.py          # Depthwise Attention Residual module
  attnres_rnn_agent.py      # RNN agent with AttnRes after GRU
  depth_mlp_rnn_agent.py    # Depth-only residual MLP control
  rnn_agent.py              # Original PyMARL RNN agent

src/config/algs/
  qmix_attnres.yaml
  qmix_attnres_l2.yaml
  qmix_attnres_block.yaml
  qmix_depth_mlp.yaml
  iql_attnres.yaml
  vdn_attnres.yaml

scripts/
  run_marl_transfer_adaptation_7gpu.sh      # Main multi-GPU experiment launcher
  summarize_marl_transfer_adaptation.py     # Summarize Sacred logs into tables
  plot_marl_transfer_curves.py              # Generate paper learning curves
  server_smoke_test.sh                      # Server-side smoke test

tests/
  test_attn_res_rnn_agent.py
  test_summarize_attnres_sc2_diagnostics.py
  test_plot_smac_results.py

paper/
  latex/                         # LaTeX paper draft source
  chinese_thesis_framework.md    # Chinese writing framework
  figures_results_checklist.md   # Figure/result checklist
```

Raw Sacred logs, StarCraft II binaries, checkpoints, and local result folders
are intentionally ignored by git.

## Environment Setup

This repository follows the original PyMARL environment style. A typical server
setup is:

```bash
cd /home/xhl009/MARL/pymarl
conda activate pymarl-sc2

export SC2PATH=/home/xhl009/MARL/pymarl/3rdparty/StarCraftII_srv
export RESULTS_DIR=/home/xhl009/MARL/pymarl-results
```

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Install StarCraft II and SMAC maps following the original PyMARL/SMAC setup:

```bash
bash install_sc2.sh
```

> Note: StarCraft II is large and is not included in this repository. Set
> `SC2PATH` to your local or server SC2 installation.

## Quick Smoke Test

Before running full experiments:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE=0 bash scripts/server_smoke_test.sh
```

## Running Experiments

### Original QMIX Baseline

```bash
python src/main.py --config=qmix --env-config=sc2 with \
  env_args.map_name=5m_vs_6m \
  seed=1 \
  local_results_path="$RESULTS_DIR" \
  use_cuda=True \
  t_max=2050000 \
  test_nepisode=32
```

### QMIX + Attention Residuals

Full four-layer AttnRes:

```bash
python src/main.py --config=qmix_attnres --env-config=sc2 with \
  env_args.map_name=5m_vs_6m \
  seed=1 \
  local_results_path="$RESULTS_DIR" \
  use_cuda=True \
  t_max=2050000 \
  test_nepisode=32
```

Lightweight two-layer AttnRes:

```bash
python src/main.py --config=qmix_attnres_l2 --env-config=sc2 with \
  env_args.map_name=5m_vs_6m \
  seed=1 \
  local_results_path="$RESULTS_DIR" \
  use_cuda=True \
  t_max=2050000 \
  test_nepisode=32
```

Depth-only control:

```bash
python src/main.py --config=qmix_depth_mlp --env-config=sc2 with \
  env_args.map_name=5m_vs_6m \
  seed=1 \
  local_results_path="$RESULTS_DIR" \
  use_cuda=True \
  t_max=2050000 \
  test_nepisode=32
```

### Multi-GPU Diagnostic Matrix

The main adaptation study can be launched with:

```bash
DRY_RUN=1 bash scripts/run_marl_transfer_adaptation_7gpu.sh
bash scripts/run_marl_transfer_adaptation_7gpu.sh
```

The launcher is designed for the local server setup used in this study. Review
GPU ids and environment variables before running it on a different machine.

## Summarizing Results

After training, generate diagnostic CSV tables:

```bash
python scripts/summarize_marl_transfer_adaptation.py \
  --sacred-dir results/sacred \
  --output-dir "$RESULTS_DIR/diagnostics"
```

Expected outputs include:

```text
marl_transfer_primary_qmix_table.csv
marl_transfer_cross_algorithm_pairs.csv
marl_transfer_cross_algorithm_aggregate.csv
marl_transfer_missing_or_partial.csv
```

The current study treats `5m_vs_6m` as the primary diagnostic map and `3s5z` as
a secondary/sanity-check map. VDN paired conclusions should not be made until
the missing VDN baseline seeds are completed.

## Plotting Paper Figures

Learning curves are generated from selected completed Sacred runs, not from all
historical runs:

```bash
python scripts/plot_marl_transfer_curves.py
```

This creates PDF figures under:

```text
paper/latex/figures/
```

The script intentionally excludes failed runs, old `2s3z` experiments, smoke
tests, and unpaired VDN candidates.

## Paper Draft

The LaTeX draft is under:

```text
paper/latex/
```

Compile it with XeLaTeX:

```bash
cd paper/latex
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
```

The generated `main.pdf` is ignored by git because the draft may not be ready
for public release. The LaTeX source, tables, and figure PDFs are intended to be
versioned.

After creating the GitHub repository, the paper can cite the code as:

```text
Code is available at: https://github.com/<user>/<repo>
```

## Current Experimental Interpretation

The current results support a cautious conclusion:

- Heavy Full/Block AttnRes variants are unstable and increase training time.
- Lightweight AttnRes-L2 shows some AUC/best-win signals, especially in early
  or cross-algorithm checks, but does not consistently improve final win rate.
- Depth-only control helps distinguish the effect of extra MLP depth from
  depth-wise attention.
- A more MARL-specific follow-up direction is agent-wise attention
  communication rather than direct depth-wise residual transfer.

## Upstream Acknowledgement

This repository is based on
[PyMARL](https://github.com/oxwhirl/pymarl), the WhiRL framework for deep
multi-agent reinforcement learning. PyMARL includes implementations of QMIX,
COMA, VDN, IQL, and QTRAN and uses
[SMAC](https://github.com/oxwhirl/smac) as its environment.

If you use this code, please also cite the original PyMARL/SMAC and algorithm
papers, including:

- SMAC: The StarCraft Multi-Agent Challenge.
- QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent
  Reinforcement Learning.
- VDN: Value-Decomposition Networks For Cooperative Multi-Agent Learning.
- IQL: Independent Q-Learning.

## Citation

Project citation placeholder:

```bibtex
@misc{attnres_smac_adaptation,
  title  = {Attention Residuals for SMAC Multi-Agent Reinforcement Learning},
  author = {Your Name},
  year   = {2026},
  note   = {Code for an empirical study of transferring Attention Residuals from LLMs to MARL agents}
}
```

SMAC citation:

```bibtex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019}
}
```

## License

This project inherits the Apache License 2.0 from PyMARL. See `LICENSE`.
