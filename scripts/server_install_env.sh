#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in PATH. Activate your conda environment first."
  exit 1
fi

echo "Python:"
python --version

echo "Installing PyTorch and torchvision..."
pip install torch torchvision

echo "Installing base dependencies..."
pip install numpy scipy pyyaml matplotlib pillow imageio pygame pytest snakeviz sacred protobuf==3.20.3

echo "Installing PySC2 and SMAC..."
pip install pysc2==3.0.0
pip install git+https://github.com/oxwhirl/smac.git

echo "Dependency installation complete."
