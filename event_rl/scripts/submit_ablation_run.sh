#!/bin/bash
#SBATCH --job-name=event_rl_test_d
#SBATCH --mem=32000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

# Test D -- recurrence ablation. Identical to Test A (submit_stnet_long_run.sh)
# except --stateless_temporal: TemporalBranch's state is reset after every
# single rollout step instead of only at episode boundaries, so no memory
# carries across steps at all. Compare this run's reward curve against Test
# A's -- similar performance means the cross-step recurrence isn't
# contributing much (each observation already spans its own 5-bin window);
# notably worse performance means the recurrence is doing real work.

module load python/3.13 2>/dev/null || true
source .venv/bin/activate

export MUJOCO_GL=egl
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
echo "=== Test D: recurrence-ablation training run (--stateless_temporal) ==="
python scripts/train.py \
    --device cuda \
    --env_id "$ENV_ID" \
    --camera_name "$CAMERA_NAME" \
    --event_source "$EVENT_SOURCE" \
    --total_timesteps 50000 \
    --checkpoint_freq 5000 \
    --max_episode_steps 300 \
    --wandb_project event_rl_test_a \
    --stateless_temporal

echo "=== job finished at $(date) ==="
