# -*- coding: utf-8 -*-
"""Extracts scalar training curves from tensorboard event files and saves
clean, presentation-ready PNG plots -- for a poster/report, not just live
tensorboard viewing in a browser.

Usage:
    # auto-discover every run under event_rl_tensorboard/, overlay them all
    python scripts/plot_training_curves.py

    # plot/compare specific runs with custom legend labels (e.g. Test A vs D)
    python scripts/plot_training_curves.py \
        --runs event_rl_tensorboard/ppo_event_123/PPO_1 event_rl_tensorboard/ppo_event_456_stateless/PPO_1 \
        --labels "Test A (recurrent)" "Test D (stateless)"
"""
import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (tag, y-axis label, plot title) -- the metrics actually discussed/watched
# throughout this project's training runs.
METRICS = [
    ("rollout/ep_len_mean", "Episode Length (steps)", "Episode Length over Training"),
    ("rollout/ep_rew_mean", "Episode Reward", "Episode Reward over Training"),
    ("train/explained_variance", "Explained Variance", "Value Function Explained Variance"),
    ("train/value_loss", "Value Loss", "Value Loss over Training"),
    ("train/entropy_loss", "Entropy Loss", "Policy Entropy over Training"),
]


def find_event_dirs(root):
    """Recursively finds every directory that directly contains a tfevents
    file -- SB3 nests these one level down (event_rl_tensorboard/<experiment_name>/PPO_1/)."""
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.startswith("events.out.tfevents") for f in filenames):
            dirs.append(dirpath)
    return sorted(dirs)


def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def plot_metric(run_dirs, labels, tag, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))
    plotted_anything = False
    for run_dir, label in zip(run_dirs, labels):
        steps, values = load_scalar(run_dir, tag)
        if steps is None:
            print(f"  (no data for {tag!r} in {run_dir} -- skipping)")
            continue
        plt.plot(steps, values, label=label, linewidth=2)
        plotted_anything = True

    if not plotted_anything:
        plt.close()
        return False

    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(run_dirs) > 1:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot poster-ready training curves from tensorboard logs")
    parser.add_argument("--logdir", type=str, default="event_rl_tensorboard",
                         help="root tensorboard directory to auto-discover runs under, if --runs isn't given")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                         help="specific run directories to plot/compare (each containing a tfevents file "
                              "directly, e.g. event_rl_tensorboard/<experiment_name>/PPO_1). Overrides "
                              "--logdir auto-discovery when given.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                         help="legend labels matching --runs, in order. Defaults to each run's parent "
                              "directory name (the experiment_name train.py generated).")
    parser.add_argument("--out_dir", type=str, default="poster_plots", help="")
    args = parser.parse_args()

    run_dirs = args.runs if args.runs else find_event_dirs(args.logdir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No tensorboard event files found under {args.logdir!r}. "
            f"Pass --logdir pointing at the right tensorboard root, or --runs with explicit paths."
        )

    labels = args.labels if args.labels else [os.path.basename(os.path.dirname(d)) for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError(f"--labels has {len(labels)} entries but --runs/discovered has {len(run_dirs)}")

    print(f"Plotting {len(run_dirs)} run(s):")
    for d, l in zip(run_dirs, labels):
        print(f"  {l!r} <- {d}")

    os.makedirs(args.out_dir, exist_ok=True)
    for tag, ylabel, title in METRICS:
        safe_name = tag.replace("/", "_")
        plot_metric(run_dirs, labels, tag, ylabel, title, os.path.join(args.out_dir, f"{safe_name}.png"))


if __name__ == "__main__":
    main()
