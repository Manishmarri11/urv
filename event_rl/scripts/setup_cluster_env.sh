#!/bin/bash
# Run this ONCE on the cluster (on a login node, no GPU needed) to build a
# dedicated venv for this project from pyproject.toml -- replaces relying on
# a shared conda env someone else maintains, so you know exactly what's
# installed. Re-run it any time pyproject.toml's dependency list changes.
set -e  # stop immediately on any real error, rather than limping on

# CONFIRMED on Unity (2026-09): `module avail python` offers up to 3.12.3,
# no 3.13 -- pyproject.toml's requires-python (>=3.12) already accommodates
# this, so no compromise needed, just the right module name. If your cluster
# differs, `module avail python` lists what's actually offered; pick the
# closest match to what pyproject.toml requires.
module load python/3.12 2>/dev/null || echo "WARNING: 'module load python/3.12' failed -- check 'module avail python' for the right name on this cluster and edit this line."

cd "$(dirname "$0")/.."   # move to event_rl/ regardless of where this was invoked from
echo "Setting up venv in: $(pwd)/.venv"

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
# Pinned to 2.5.1/cu124 specifically for broad GPU compatibility -- PyTorch
# dropped Maxwell (compute capability 5.x, e.g. GTX Titan X) kernel support
# starting with 2.8.0, and the cluster can hand us any GPU in the pool, so we
# can't assume a specific card. Installed explicitly (not left to plain
# `pip install torch`) so this exact build is what lands, not whatever's
# newest on PyPI that day. If preflight_check.py's CUDA check still fails
# after this, it's a driver/allocation issue, not a "wrong torch build" one.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .

echo ""
echo "=== Install complete. Verifying: ==="
python -c "
import torch, gymnasium, stable_baselines3, mujoco, wandb, PIL
print('torch', torch.__version__, '(cuda build:', torch.version.cuda, ')')
print('gymnasium', gymnasium.__version__)
print('stable_baselines3', stable_baselines3.__version__)
print('mujoco', mujoco.__version__)
print('wandb', wandb.__version__)
print('PIL', PIL.__version__)
print('ALL IMPORTS OK')
"
echo ""
echo "Done. Activate later with: source $(pwd)/.venv/bin/activate"
