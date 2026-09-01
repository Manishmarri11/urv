#!/bin/bash
#SBATCH --job-name=event_rl_dual_camera
#SBATCH --mem=32000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

# Dual-camera run -- same real 50,000-step training config as
# submit_poster_run.sh (VecNormalize fix included), but against your
# environment teammate's CartpoleWorldDualCamera-v0 (2D tilt/cart-correction,
# cliff-path terrain) instead of the original single-camera/single-axis env.
# camera_name=pole is the true egocentric feed your teammate intends for the
# event pipeline (not "world", which is a 3rd-person convenience view).
# Auto-produces poster-ready PNG plots at the end, same as submit_poster_run.sh.

module load python/3.13 2>/dev/null || true
source .venv/bin/activate

export MUJOCO_GL=egl
export WANDB_MODE=offline

ENV_ID=CartpoleWorldDualCamera-v0
CAMERA_NAME=pole

echo "=== job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

echo ""
echo "=== pre-flight check (env_id=$ENV_ID, camera_name=$CAMERA_NAME) ==="
python scripts/preflight_check.py --env_id "$ENV_ID" --camera_name "$CAMERA_NAME"
PREFLIGHT_STATUS=$?
if [ $PREFLIGHT_STATUS -ne 0 ]; then
    echo "=== PRE-FLIGHT FAILED (exit $PREFLIGHT_STATUS) -- aborting before training. ==="
    exit $PREFLIGHT_STATUS
fi

echo ""
echo "=== Training run ==="
TRAIN_LOG=$(mktemp)
python scripts/train.py \
    --device cuda \
    --env_id "$ENV_ID" \
    --camera_name "$CAMERA_NAME" \
    --total_timesteps 50000 \
    --checkpoint_freq 5000 \
    --max_episode_steps 300 \
    --wandb_project event_rl_dual_camera \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "=== TRAINING FAILED (exit $TRAIN_STATUS) -- skipping plot generation. ==="
    rm -f "$TRAIN_LOG"
    exit $TRAIN_STATUS
fi

# Same portable grep -oE + sed extraction as submit_poster_run.sh -- grep -P
# (lookbehind) confirmed to fail outright on some locales.
EXPERIMENT_NAME=$(grep -oE 'Experiment name: .*' "$TRAIN_LOG" | tail -1 | sed 's/^Experiment name: //')
rm -f "$TRAIN_LOG"

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "=== WARNING: could not find 'Experiment name:' in training output -- skipping auto-plot. ==="
    echo "=== Run scripts/plot_training_curves.py manually once you find the right experiment_name. ==="
else
    echo ""
    echo "=== Plotting: $EXPERIMENT_NAME ==="
    python scripts/plot_training_curves.py \
        --runs "event_rl_tensorboard/${EXPERIMENT_NAME}/PPO_1" \
        --labels "STNet + PPO (dual-camera)" \
        --out_dir poster_plots_dual_camera
fi

echo "=== job finished at $(date) ==="
