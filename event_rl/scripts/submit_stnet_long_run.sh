#!/bin/bash
#SBATCH --job-name=event_rl_test_a
#SBATCH --mem=32000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_%j.out

# Test A -- the main real-training test. Same environment/converter as the
# smoke test, but ~24 PPO update cycles (50,000 / n_steps=2048) instead of
# the smoke test's 32 (2048 / n_steps=64) -- enough to see whether ep_rew_mean
# actually breaks away from the ~5 random baseline. Time budget: measured
# ~10-11 fps sustained on the A16 during the smoke test -> 50,000 steps is
# roughly 80-90 min of compute; 03:00:00 leaves buffer for queueing/startup/
# checkpoint I/O. Adjust --time down if your cluster charges heavily for
# unused wall-clock, or up if you raise --total_timesteps below.

module load python/3.13 2>/dev/null || true
source .venv/bin/activate

export MUJOCO_GL=egl
# WANDB_MODE=offline: still writes local run files (dashboard-viewable via
# `wandb sync` later) without requiring a login/network call from inside an
# unattended batch job. TensorBoard logging (train.py's tensorboard_log=...)
# happens unconditionally regardless of wandb, so that's the "can't fail" log.
export WANDB_MODE=offline

echo "=== job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

echo ""
echo "=== pre-flight check ==="
python scripts/preflight_check.py
PREFLIGHT_STATUS=$?
if [ $PREFLIGHT_STATUS -ne 0 ]; then
    echo "=== PRE-FLIGHT FAILED (exit $PREFLIGHT_STATUS) -- aborting before training. ==="
    exit $PREFLIGHT_STATUS
fi

echo ""
echo "=== Test A: real STNet training run ==="
python scripts/train.py \
    --device cuda \
    --total_timesteps 50000 \
    --checkpoint_freq 5000 \
    --max_episode_steps 300 \
    --wandb_project event_rl_test_a

echo "=== job finished at $(date) ==="
