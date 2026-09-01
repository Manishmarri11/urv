#!/bin/bash
#SBATCH --job-name=event_rl_poster
#SBATCH --mem=32000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

# Poster run -- real 50,000-step training (same config as submit_stnet_long_run.sh,
# including the VecNormalize fix) that automatically produces poster-ready PNG
# plots at the end, so there's no separate manual plotting step afterward.
# Pull slurm_<jobid>.out's "Experiment name: ..." line if you need to
# re-plot or find this run's checkpoints later.

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
echo "=== Training run ==="
TRAIN_LOG=$(mktemp)
python scripts/train.py \
    --device cuda \
    --total_timesteps 50000 \
    --checkpoint_freq 5000 \
    --max_episode_steps 300 \
    --wandb_project event_rl_poster \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "=== TRAINING FAILED (exit $TRAIN_STATUS) -- skipping plot generation. ==="
    rm -f "$TRAIN_LOG"
    exit $TRAIN_STATUS
fi

# Extract the exact experiment name train.py generated (timestamp-based, not
# knowable in advance) so plotting targets ONLY this run, not every run ever
# logged under event_rl_tensorboard/. grep -oE + sed instead of grep -P
# (lookbehind) -- confirmed locally that -P fails outright on some locales
# ("supports only unibyte and UTF-8 locales"), this form doesn't depend on
# PCRE support being compiled into grep at all.
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
        --labels "STNet + PPO" \
        --out_dir poster_plots
fi

echo "=== job finished at $(date) ==="
