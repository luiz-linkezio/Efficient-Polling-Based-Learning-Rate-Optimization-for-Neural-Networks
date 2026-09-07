"""Redraw the paper's figures from the runs recorded by ``cifar10.py``.

Reads every ``results/cifar10/<method>_seed<N>.json`` and plots the mean curve
across seeds with a shaded min-max band, so a figure can never disagree with
the table: both are computed from the same files.

    python examples/plot_results.py --results-dir results/cifar10 --out-dir images

Requires the ``examples`` extra: ``pip install "efficient-polling-lr-scheduler[examples]"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Method order and labels follow the table in cifar10.py.
LABELS = {
    "baseline": "SGD (fixed 1e-3)",
    "polling": "Polling (base paper)",
    "efficient": "Efficient Polling (ours)",
    "adam": "Adam (1e-3)",
    "cosine": "SGD + cosine annealing",
    "step": "SGD + step decay",
    "plateau": "SGD + ReduceLROnPlateau",
    "sps": "SPS (Polyak)",
    "armijo": "Armijo line search",
    "efficient_fixed": "Ablation: fixed interval",
    "efficient_random": "Ablation: random trigger",
    "efficient_relative": "Efficient Relative Polling (ours, per batch)",
    "efficient_relative_epoch": "Efficient Relative Polling (ours, per epoch)",
}


def load(results_dir: Path) -> dict[str, list[dict]]:
    """Group the result files by method, ordered as in ``LABELS``."""
    runs: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*_seed*.json")):
        record = json.loads(path.read_text())
        runs.setdefault(record["method"], []).append(record)
    return {m: runs[m] for m in LABELS if m in runs}


def curves(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, min and max of one history key across the seeds of a method."""
    stacked = np.asarray([r["history"][key] for r in runs], dtype=float)
    return stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0)


def plot_losses(runs_by_method: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.set_title("Loss per epoch (mean over seeds, band = min-max)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    for index, (method, runs) in enumerate(runs_by_method.items()):
        color = f"C{index}"
        epochs = np.arange(1, len(runs[0]["history"]["train_loss"]) + 1)

        train_mean, train_lo, train_hi = curves(runs, "train_loss")
        val_mean, val_lo, val_hi = curves(runs, "val_loss")

        # Both curves are recorded on the same convention for every method: the
        # training loss is the batch loss at the point the gradient was taken.
        ax.plot(
            epochs,
            train_mean,
            color=color,
            lw=1.5,
            ls="--",
            alpha=0.6,
            label=f"{LABELS[method]} train",
        )
        ax.plot(epochs, val_mean, color=color, lw=1.5, label=f"{LABELS[method]} val")
        ax.fill_between(epochs, train_lo, train_hi, color=color, alpha=0.12, lw=0)
        ax.fill_between(epochs, val_lo, val_hi, color=color, alpha=0.12, lw=0)

    ax.set_xlim(0.5, len(epochs) + 0.5)
    everything = np.concatenate(
        [np.asarray(r["history"]["val_loss"]) for runs in runs_by_method.values() for r in runs]
    )
    ax.set_ylim(0, float(np.percentile(everything[np.isfinite(everything)], 98)) * 1.15)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_learning_rates(runs_by_method: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.set_title("Mean learning rate per epoch (mean over seeds, band = min-max)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.grid(True, which="both", alpha=0.3)

    for index, (method, runs) in enumerate(runs_by_method.items()):
        color = f"C{index}"
        epochs = np.arange(1, len(runs[0]["history"]["lr"]) + 1)
        mean, lo, hi = curves(runs, "lr")
        ax.plot(epochs, mean, color=color, lw=1.5, label=LABELS[method])
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.15, lw=0)

    ax.set_xlim(0.5, len(epochs) + 0.5)
    ax.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_polls(runs_by_method: dict[str, list[dict]], out_path: Path) -> None:
    runs = runs_by_method.get("efficient")
    if not runs:
        print("no efficient-polling runs found, skipping the poll figure")
        return

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.set_title("Efficient Polling: polls per epoch (mean over seeds, band = min-max)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Polls")
    ax.grid(True, alpha=0.3)

    epochs = np.arange(1, len(runs[0]["history"]["polls"]) + 1)
    mean, lo, hi = curves(runs, "polls")
    ax.plot(epochs, mean, color="C2", lw=1.5)
    ax.fill_between(epochs, lo, hi, color="C2", alpha=0.2, lw=0)
    ax.set_xlim(0.5, len(epochs) + 0.5)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results/cifar10")
    parser.add_argument("--out-dir", default="images")
    args = parser.parse_args()

    runs_by_method = load(Path(args.results_dir))
    if not runs_by_method:
        raise SystemExit(f"no result files in {args.results_dir}; run examples/cifar10.py first")
    for method, runs in runs_by_method.items():
        print(f"{method}: {len(runs)} seed(s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_losses(runs_by_method, out_dir / "training_comparison_losses.png")
    plot_learning_rates(runs_by_method, out_dir / "training_comparison_LRs.png")
    plot_polls(runs_by_method, out_dir / "polls_per_epoch.png")


if __name__ == "__main__":
    main()
