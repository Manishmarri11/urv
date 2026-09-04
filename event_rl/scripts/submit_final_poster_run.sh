#!/bin/bash
#SBATCH --job-name=event_rl_final_poster
#SBATCH --mem=32000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=05:00:00
#SBATCH --output=slurm_%j.out

# FINAL POSTER RUN -- one job that produces everything the poster needs:
#   1. pre-flight (fails in seconds if the cluster env is wrong, not 3h in)
#   2. the real 50,000-step training run
#   3. all training-curve PNGs, including the seven pole-angle statistics
#   4. videos from an EARLY and the FINAL checkpoint, for the visual
#      "does it hold the pole up longer by the end" comparison
#
# Config matches submit_v2e_run.sh exactly (dual-camera env, egocentric pole
# camera, real v2e DVS simulation at densified 0.05 thresholds, event_stillness
# reward shaping at the coefficient calibrated for those thresholds). Every
# value is passed EXPLICITLY -- train.py's defaults have silently changed once
# already, and a final run is the worst place to inherit a surprise.
#
# TIME BUDGET: measured ~4.35 env-steps/s on this config, so 50,000 steps is
# ~3.2h, plus pre-flight and two video renders. 05:00:00 rather than 04:00:00
# because an earlier run was cut off one iteration short of finishing at the
# wire, and that is a far more expensive mistake on a final run than the
# unused allocation is.
#
# READ BEFORE INTERPRETING THE RESULTS (all measured, see README):
#   - This env's RANDOM-POLICY baseline is ~150 steps, NOT the ~5 of the old
#     single-camera env (gear=5 is 20x weaker, so the pole falls slowly).
#     Iteration 1 of the log IS your baseline -- compare against that, not
#     against the old runs' numbers.
#   - ep_rew_mean is NOT comparable to earlier unshaped runs: the shaping
#     penalty makes it sit below ep_len_mean. ep_len_mean and the angle
#     statistics are the metrics that stay comparable.
#   - The shaping penalizes ALL event activity, which on the pole-mounted
#     camera includes the cart motion the agent needs in order to correct.
#     If the curve stays flat, suspect that before suspecting the encoder.

module load python/3.12 2>/dev/null || true  # 3.12 = what setup_cluster_env.sh builds the venv with (Unity has no 3.13)
source .venv/bin/activate

export MUJOCO_GL=egl
export WANDB_MODE=offline

ENV_ID=CartpoleWorldDualCamera-v0
CAMERA_NAME=pole
EVENT_SOURCE=v2e
V2E_POS_THRES=0.05
V2E_NEG_THRES=0.05
REWARD_SHAPING=event_stillness
EVENT_SHAPING_COEF=2.0
TOTAL_TIMESTEPS=50000
CHECKPOINT_FREQ=5000
MAX_EPISODE_STEPS=300
EARLY_CHECKPOINT_STEPS=5000   # the "before" half of the early-vs-late video pair
VIDEO_EPISODES=5

echo "=== job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

echo ""
echo "=== [1/4] pre-flight (env_id=$ENV_ID, camera=$CAMERA_NAME, event_source=$EVENT_SOURCE, thres=$V2E_POS_THRES/$V2E_NEG_THRES, shaping=$REWARD_SHAPING) ==="
python scripts/preflight_check.py \
    --env_id "$ENV_ID" \
    --camera_name "$CAMERA_NAME" \
    --event_source "$EVENT_SOURCE" \
    --v2e_pos_thres "$V2E_POS_THRES" \
    --v2e_neg_thres "$V2E_NEG_THRES" \
    --reward_shaping "$REWARD_SHAPING" \
    --event_shaping_coef "$EVENT_SHAPING_COEF"
PREFLIGHT_STATUS=$?
if [ $PREFLIGHT_STATUS -ne 0 ]; then
    echo "=== PRE-FLIGHT FAILED (exit $PREFLIGHT_STATUS) -- aborting before training. ==="
    exit $PREFLIGHT_STATUS
fi

echo ""
echo "=== [2/4] training ($TOTAL_TIMESTEPS steps) ==="
TRAIN_LOG=$(mktemp)
python scripts/train.py \
    --device cuda \
    --env_id "$ENV_ID" \
    --camera_name "$CAMERA_NAME" \
    --event_source "$EVENT_SOURCE" \
    --v2e_pos_thres "$V2E_POS_THRES" \
    --v2e_neg_thres "$V2E_NEG_THRES" \
    --reward_shaping "$REWARD_SHAPING" \
    --event_shaping_coef "$EVENT_SHAPING_COEF" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --checkpoint_freq "$CHECKPOINT_FREQ" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --wandb_project event_rl_final_poster \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "=== TRAINING FAILED (exit $TRAIN_STATUS) -- skipping plots and videos. ==="
    rm -f "$TRAIN_LOG"
    exit $TRAIN_STATUS
fi

# grep -oE + sed rather than grep -P: -P (lookbehind) fails outright on some
# locales ("supports only unibyte and UTF-8 locales") and this form needs no
# PCRE support compiled into grep at all.
EXPERIMENT_NAME=$(grep -oE 'Experiment name: .*' "$TRAIN_LOG" | tail -1 | sed 's/^Experiment name: //')
rm -f "$TRAIN_LOG"

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "=== WARNING: could not recover the experiment name from the training output. ==="
    echo "=== Training DID succeed -- run plot_training_curves.py and render_checkpoint_video.py"
    echo "=== manually against the newest directory in event_rl_tensorboard/ and saved_models/. ==="
    echo "=== job finished at $(date) ==="
    exit 0
fi

echo ""
echo "=== [3/4] plots for: $EXPERIMENT_NAME ==="
# Produces every metric in plot_training_curves.py's METRICS list, which now
# includes the seven angle/tilt_abs_deg_* statistics alongside the reward,
# value-function and entropy curves.
python scripts/plot_training_curves.py \
    --runs "event_rl_tensorboard/${EXPERIMENT_NAME}/PPO_1" \
    --labels "STNet + PPO (v2e, shaped)" \
    --out_dir poster_plots_final \
    || echo "WARNING: plotting failed -- training results are still intact, re-run plot_training_curves.py by hand."

echo ""
echo "=== [4/4] early-vs-late videos ==="
# --camera_name is deliberately the SAME "pole" camera used for training.
# render_checkpoint_video.py builds one env whose camera feeds BOTH the video
# frames and the policy's observations, so rendering with "world" would give a
# prettier third-person clip while silently evaluating the policy on an
# observation distribution it never trained on. Correct evaluation wins here;
# the clip is egocentric as a result.
#
# --reward_shaping is deliberately NOT passed below, so it defaults to "none"
# and the per-episode summary reports the GROUND-TRUTH survival reward. That
# is what you want when judging "did it actually hold the pole up longer" --
# adding the shaping flag here would subtract the motion penalty and make the
# printed reward incomparable to the earlier runs. Not an oversight.
render_checkpoint () {
    local checkpoint_path="$1"
    local label="$2"
    if [ ! -f "$checkpoint_path" ]; then
        echo "  (skipping $label -- $checkpoint_path not found)"
        return 0
    fi
    echo "  rendering $label: $checkpoint_path"
    # Never fatal: training and plots are already saved by this point, and a
    # render failure should not cost the whole job's results.
    python scripts/render_checkpoint_video.py \
        --checkpoint "$checkpoint_path" \
        --env_id "$ENV_ID" \
        --camera_name "$CAMERA_NAME" \
        --event_source "$EVENT_SOURCE" \
        --v2e_pos_thres "$V2E_POS_THRES" \
        --v2e_neg_thres "$V2E_NEG_THRES" \
        --max_episode_steps "$MAX_EPISODE_STEPS" \
        --n_episodes "$VIDEO_EPISODES" \
        --device cuda \
        --out_dir videos_final \
        || echo "  WARNING: rendering $label failed -- continuing."
}

render_checkpoint "saved_models/${EXPERIMENT_NAME}_${EARLY_CHECKPOINT_STEPS}_steps.zip" "EARLY (${EARLY_CHECKPOINT_STEPS} steps)"
render_checkpoint "saved_models/${EXPERIMENT_NAME}.zip" "FINAL (${TOTAL_TIMESTEPS} steps)"

echo ""
echo "=== done. Poster artifacts: ==="
echo "  plots  : poster_plots_final/"
echo "  videos : videos_final/"
echo "  run    : $EXPERIMENT_NAME"
echo "=== job finished at $(date) ==="
