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
echo "=== Test D: recurrence-ablation training run (--stateless_temporal) ==="
python scripts/train.py \
    --device cuda \
    --total_timesteps 50000 \
    --checkpoint_freq 5000 \
    --max_episode_steps 300 \
    --wandb_project event_rl_test_a \
    --stateless_temporal

echo "=== job finished at $(date) ==="
