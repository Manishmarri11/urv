#!/bin/bash
#SBATCH --job-name=event_rl_test_a
#SBATCH --mem=32000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

# Test A -- the main real-training test. Same environment/converter as the
# smoke test, but ~24 PPO update cycles (50,000 / n_steps=2048) instead of
# the smoke test's 32 (2048 / n_steps=64) -- enough to see whether ep_rew_mean
# actually breaks away from the ~5 random baseline. Time budget: the first
# real run measured ~2.94 hours actual (fps drops to ~4-5 sustained once
# training-phase compute -- n_epochs=10 x n_steps/batch_size=32 minibatches
# per iteration -- dominates wall-clock, not just rollout collection) and got
# cut off by an earlier 03:00:00 limit one iteration short of finishing.
# 04:00:00 gives real margin instead of finishing right at the wire.

module load python/3.12 2>/dev/null || true  # 3.12 = what setup_cluster_env.sh builds the venv with (Unity has no 3.13)
source .venv/bin/activate

export MUJOCO_GL=egl
# WANDB_MODE=offline: still writes local run files (dashboard-viewable via
# `wandb sync` later) without requiring a login/network call from inside an
# unattended batch job. TensorBoard logging (train.py's tensorboard_log=...)
# happens unconditionally regardless of wandb, so that's the "can't fail" log.
export WANDB_MODE=offline

# Env/camera/converter are passed EXPLICITLY below rather than relying on
# train.py's defaults -- those defaults have changed once already
# (single-camera "side" -> dual-camera "pole"), which silently changed what
# this script ran without any edit to the script itself. Test A and Test D
# must agree on all three for the comparison to be apples-to-apples.
ENV_ID=CartpoleWorldDualCamera-v0
CAMERA_NAME=pole
EVENT_SOURCE=fake

echo "=== job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

echo ""
echo "=== pre-flight check (env_id=$ENV_ID, camera_name=$CAMERA_NAME, event_source=$EVENT_SOURCE) ==="
python scripts/preflight_check.py \
    --env_id "$ENV_ID" \
    --camera_name "$CAMERA_NAME" \
    --event_source "$EVENT_SOURCE"
PREFLIGHT_STATUS=$?
if [ $PREFLIGHT_STATUS -ne 0 ]; then
    echo "=== PRE-FLIGHT FAILED (exit $PREFLIGHT_STATUS) -- aborting before training. ==="
    exit $PREFLIGHT_STATUS
fi

echo ""
echo "=== Test A: real STNet training run ==="
python scripts/train.py \
    --device cuda \
    --env_id "$ENV_ID" \
    --camera_name "$CAMERA_NAME" \
    --event_source "$EVENT_SOURCE" \
    --total_timesteps 50000 \
    --checkpoint_freq 5000 \
    --max_episode_steps 300 \
    --wandb_project event_rl_test_a

echo "=== job finished at $(date) ==="
