#!/bin/bash
#SBATCH --job-name=event_rl_v2e
#SBATCH --mem=32000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

# v2e run -- identical to submit_poster_run.sh except --event_source v2e:
# the real, stateful DVS camera simulation (per-pixel threshold mismatch,
# leak/shot noise, sub-frame-resolved timestamps) instead of the "fake"
# two-frame-diff placeholder every earlier successful run trained on.
#
# MEASURED BEFORE RUNNING THIS (see README's "Event source" section): at v2e's
# default thresholds the observations are ~17x sparser than "fake" (~0.2% vs
# ~3.4% nonzero pixels), with occasional fully-blank steps. That is real DVS
# behavior, not a bug, but it IS a materially different input distribution --
# so treat this as an experiment whose outcome is genuinely unknown, and keep
# submit_poster_run.sh's "fake" run as the comparable baseline. If this fails
# to learn, --v2e_pos_thres/--v2e_neg_thres below 0.2 densify the stream.
#
# Env/camera/converter are passed EXPLICITLY rather than relying on train.py's
# defaults -- those defaults have changed once already, silently changing what
# scripts like this ran without any edit to the script itself.

module load python/3.13 2>/dev/null || true
source .venv/bin/activate

export MUJOCO_GL=egl
export WANDB_MODE=offline

ENV_ID=CartpoleWorldDualCamera-v0
CAMERA_NAME=pole
EVENT_SOURCE=v2e

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
echo "=== Training run (real v2e DVS simulation) ==="
TRAIN_LOG=$(mktemp)
python scripts/train.py \
    --device cuda \
    --env_id "$ENV_ID" \
    --camera_name "$CAMERA_NAME" \
    --event_source "$EVENT_SOURCE" \
    --total_timesteps 50000 \
    --checkpoint_freq 5000 \
    --max_episode_steps 300 \
    --wandb_project event_rl_v2e \
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
        --labels "STNet + PPO (v2e)" \
        --out_dir poster_plots_v2e
fi

echo "=== job finished at $(date) ==="
