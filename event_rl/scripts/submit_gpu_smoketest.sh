#!/bin/bash
#SBATCH --job-name=event_rl_smoketest
#SBATCH --mem=32000
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.out

# #SBATCH resource requests mirror URV2/event/main_event.sh's conventions
# (same partition name, memory, GPU/node request style) so this fits your
# existing cluster's setup rather than guessing unfamiliar ones. ADJUST if
# your account/partition differs from that.
#
# Environment: a dedicated venv built from pyproject.toml (run
# scripts/setup_cluster_env.sh once first), NOT the shared URV2 'event'
# conda env -- this way you know exactly what's installed, rather than
# relying on someone else's shared environment having everything this
# project needs (it was missing wandb and pillow, found by inspection).

module load python/3.13 2>/dev/null || true  # match whatever setup_cluster_env.sh used
# Relative to CWD, consistent with scripts/preflight_check.py and
# scripts/train.py below -- this whole script assumes `sbatch` was run from
# inside event_rl/, same as setup_cluster_env.sh's instructions.
source .venv/bin/activate

# Headless GPU rendering for MuJoCo -- there's no display on a compute node.
# EGL gives GPU-accelerated headless rendering; if EGL isn't available on
# this cluster's driver setup, fall back to osmesa (CPU rasterization,
# slower but functional, still headless-safe) by commenting the egl line
# and uncommenting the osmesa one instead.
export MUJOCO_GL=egl
# export MUJOCO_GL=osmesa

echo "=== job started on $(hostname) at $(date) ==="
echo "=== GPU visible to this job: ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

echo ""
echo "=== Step 1: pre-flight check (fails fast, before wasting GPU time) ==="
python scripts/preflight_check.py
PREFLIGHT_STATUS=$?
if [ $PREFLIGHT_STATUS -ne 0 ]; then
    echo "=== PRE-FLIGHT FAILED (exit $PREFLIGHT_STATUS) -- aborting before training. Check the output above. ==="
    exit $PREFLIGHT_STATUS
fi

echo ""
echo "=== Step 2: bounded smoke-test training run on real GPU ==="
# Bounded on purpose -- this is "does it work and how fast is it," not a
# real production run. Once this confirms healthy, total_timesteps can be
# raised for a real run (or wandb/tensorboard used to watch a long one).
WANDB_MODE=disabled python scripts/train.py \
    --device cuda \
    --total_timesteps 2048 \
    --n_steps 64 \
    --batch_size 32 \
    --n_epochs 4 \
    --max_episode_steps 100 \
    --wandb_project event_rl_gpu_smoketest

echo "=== job finished at $(date) ==="
