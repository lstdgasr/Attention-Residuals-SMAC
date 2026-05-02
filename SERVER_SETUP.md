# PyMARL / SMAC Server Setup

This guide is for a Linux GPU server with conda installed.

## Recommended layout

- Project: `/data/$USER/marl/pymarl`
- Results: `/data/$USER/pymarl-results`
- SC2: `/data/$USER/marl/pymarl/3rdparty/StarCraftII`
- Conda env: `pymarl-sc2`

## 1. Upload the project

Prefer uploading this already-tested local copy instead of cloning the upstream repo from scratch.

Example from your local machine:

```bash
scp -r D:/MARL/pymarl your_name@server:/data/$USER/marl/
```

After login:

```bash
cd /data/$USER/marl/pymarl
```

## 2. Create the environment

```bash
conda create -y -n pymarl-sc2 python=3.8
conda activate pymarl-sc2
```

## 3. Install dependencies

You can use the helper script:

```bash
bash scripts/server_install_env.sh
```

Or install manually:

```bash
pip install torch torchvision
pip install numpy scipy pyyaml matplotlib pillow imageio pygame pytest snakeviz sacred protobuf==3.20.3
pip install pysc2==3.0.0
pip install git+https://github.com/oxwhirl/smac.git
```

If `git+https://...` is unstable on the server, clone `smac` locally and install it from disk.

## 4. Install SC2 and maps

```bash
export EXP_DIR=/data/$USER/marl
cd /data/$USER/marl/pymarl
bash install_sc2.sh
```

Then export `SC2PATH`:

```bash
export SC2PATH=/data/$USER/marl/pymarl/3rdparty/StarCraftII
```

Persist it:

```bash
echo 'export SC2PATH=/data/'"$USER"'/marl/pymarl/3rdparty/StarCraftII' >> ~/.bashrc
source ~/.bashrc
```

## 5. Verify the environment

Quick checks:

```bash
python -c "import torch, pysc2, smac"
echo $SC2PATH
nvidia-smi
ls $SC2PATH
ls $SC2PATH/Maps/SMAC_Maps | head
```

## 6. Smoke test

```bash
cd /data/$USER/marl/pymarl
bash scripts/server_smoke_test.sh
```

This runs a short `QMIX + 3m` validation job to confirm the full training loop works.

## 7. Formal training

Start a tmux session first:

```bash
tmux new -s pymarl
conda activate pymarl-sc2
cd /data/$USER/marl/pymarl
```

Run the baseline:

```bash
bash scripts/train_qmix_2s3z.sh
```

Detach with `Ctrl+B` then `D`, and return later with:

```bash
tmux attach -t pymarl
```

## 8. Multi-seed runs

Run seeds one by one:

```bash
bash scripts/run_qmix_multiseed.sh
```

Edit the seed list in that script if you want 3, 5, or more seeds.

## 9. What to keep for paper / CV usage

Retain these items for every run:

- map name
- algorithm
- seed
- SC2 version
- GPU model
- full command
- `results/sacred/*/config.json`
- `results/sacred/*/metrics.json`

## Notes

- The upstream README notes that SMAC paper results used `SC2.4.6.2.69232`, not `SC2.4.10`.
- Running successfully on a newer SC2 version is still useful, but it is not a paper-faithful numerical comparison by default.
- The smoke test is not the formal reproduction; it is only an environment validation step.
