# PyMARL / SMAC Local Setup

This repo is prepared for a Windows-first smoke test with data and outputs on `D:`.

## Paths

- Repo: `D:\MARL\pymarl`
- Results: `D:\MARL\pymarl-results`
- SC2: `D:\StarCraft II`
- Conda env: `D:\Anaconda\envs\pymarl-sc2`

## What changed

- `src/main.py` was updated for Python 3.8+ compatibility:
  - use `collections.abc.Mapping`
  - use `yaml.safe_load`
- `run_local_smoke.ps1` runs a short CPU smoke test and writes results to `D:\MARL\pymarl-results`

## Local bring-up

Create the environment:

```powershell
conda create -y -p D:\Anaconda\envs\pymarl-sc2 python=3.8
```

Activate it:

```powershell
conda activate D:\Anaconda\envs\pymarl-sc2
```

Install dependencies incrementally so compatibility issues are easier to isolate:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy scipy pyyaml matplotlib pillow imageio pygame pytest snakeviz tensorboard-logger jsonpickle==0.9.6 sacred==0.8.4
python -m pip install pysc2==3.0.0 protobuf==3.20.3
python -m pip install git+https://github.com/oxwhirl/smac.git
```

If GPU inference/training fails, keep `use_cuda=False` for the smoke test and debug CUDA separately.

## Maps

SMAC maps must exist under:

```text
D:\StarCraft II\Maps\SMAC_Maps
```

## Smoke test

```powershell
conda activate D:\Anaconda\envs\pymarl-sc2
cd D:\MARL\pymarl
.\run_local_smoke.ps1
```

## Server fallback

For longer or more faithful reproduction, prefer a Linux server with a conda environment and a separate SC2/SMAC install.
