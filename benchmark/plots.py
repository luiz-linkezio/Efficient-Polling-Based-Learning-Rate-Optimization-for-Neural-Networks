"""The paper's figures, drawn from the recorded runs.

Each curve is the mean over seeds with a min-max band, read from
``results/<dataset>/``, so a figure can never disagree with the table. The
figures of a dataset go to ``images/<dataset>/`` under the names the paper cites:

    python -m benchmark.plots                    # CIFAR-10
    python -m benchmark.plots --dataset mnist

Needs matplotlib, from the ``benchmark`` extra.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure

from .datasets import DATASETS, spec_for
from .methods import LABELS, EfficientPolling
from .sweep import REPO_DIR, load_initial_lr_runs, load_runs

__all__ = [
    "METHOD_COLORS",
    "METHOD_GROUPS",
    "METHOD_STYLES",
    "animate_loss_overlay",
    "animate_lr_overlay",
    "figures_dir",
    "loss_comparison",
    "lr_comparison",
    "lr_robustness",
    "polls_per_epoch",
    "save_figures",
]

Runs = dict[str, list[dict[str, Any]]]
Groups = Sequence[tuple[str, Sequence[str]]]

# --- vocabulary: one color per method, fixed and never cycled -----------------
# Each method keeps its color in every figure. The palette was checked for
# colorblind separation panel by panel rather than picked by eye, and the line
# style is a second encoding, so the panels survive a grayscale print.
METHOD_COLORS = {
    "baseline": "#0072B2",
    "adam": "#D55E00",
    "cosine": "#009E73",
    "step": "#CC79A7",
    "plateau": "#56B4E9",
    "sps": "#E69F00",
    "armijo": "#7570B3",
    "polling": "#E7298A",
    "efficient": "#1B7837",
    "efficient_fixed": "#4575B4",
    "efficient_random": "#BF812D",
    "efficient_relative": "#B2182B",
    "efficient_relative_epoch": "#01665E",
}

METHOD_STYLES: dict[str, Any] = {
    "baseline": "-",
    "adam": "--",
    "cosine": "-.",
    "step": ":",
    "plateau": (0, (3, 1, 1, 1)),
    "sps": "--",
    "armijo": "-.",
    "polling": ":",
    "efficient": "-",
    "efficient_fixed": "--",
    "efficient_random": "-.",
    "efficient_relative": "-",
    "efficient_relative_epoch": "--",
}

# Short names, for legends that have to fit a 2-inch panel.
SHORT_LABELS = {
    "baseline": "SGD 1e-3",
    "adam": "Adam",
    "cosine": "cosine",
    "step": "step decay",
    "plateau": "plateau",
    "sps": "SPS",
    "armijo": "Armijo",
    "polling": "Polling",
    "efficient": "Efficient",
    "efficient_fixed": "abl. fixed",
    "efficient_random": "abl. random",
    "efficient_relative": "Eff. relative",
    "efficient_relative_epoch": "Eff. rel. epoch",
}

# Thirteen curves on one axes are unreadable, so every figure is small multiples:
# one panel per family, at most five series each. `efficient` repeats in the last
# two panels because it is the reference both of them are read against.
METHOD_GROUPS: Groups = (
    ("Fixed rate and schedulers", ("baseline", "adam", "cosine", "step", "plateau")),
    ("Step size measured on the batch", ("sps", "armijo", "polling", "efficient")),
    ("Trigger ablation", ("efficient", "efficient_fixed", "efficient_random")),
    ("Efficient relative polling", ("efficient", "efficient_relative", "efficient_relative_epoch")),
)

# The figures are authored at the size they are printed at, so nothing is
# downscaled into illegibility by \includegraphics.
PAPER_RC = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.titlesize": 9,
}
COLUMN_SIZE = (3.45, 2.35)  # one IEEE column, single panel


def column_stack(n_panels: int) -> tuple[float, float]:
    """One column wide, panels stacked: the figures never interrupt the paper's two columns."""
    return (3.45, 1.75 * n_panels)


def figures_dir(dataset: str) -> Path:
    """Where a dataset's figures go: one folder per dataset, as for the records."""
    return REPO_DIR / "images" / spec_for(dataset).key


# --- reading curves out of the records ----------------------------------------


def mean_history(runs: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Element-wise mean over seeds of every per-epoch curve."""
    history = runs[0]["history"]
    return {
        key: np.mean([r["history"][key] for r in runs], axis=0).tolist()
        for key in history
        if isinstance(history[key], list)
    }


def band(runs: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Min and max across seeds of one curve."""
    stacked = np.asarray([r["history"][key] for r in runs], dtype=float)
    return stacked.min(axis=0), stacked.max(axis=0)


def resolve_groups(runs_per_method: Runs, groups: Groups) -> list[tuple[str, list[str]]]:
    """Drop methods with no runs on disk, then panels left empty."""
    resolved = [
        (title, [m for m in methods if runs_per_method.get(m)]) for title, methods in groups
    ]
    return [(title, methods) for title, methods in resolved if methods]


def divergence(runs: list[dict[str, Any]], key: str = "val_loss") -> tuple[int | None, int]:
    """First epoch whose loss stops being finite, and how many seeds got there.

    A mean curve inherits the NaN of a diverged seed and simply stops, which
    would read as missing data unless the figure says otherwise, so the
    figures mark the break and count it in the legend.
    """
    firsts = []
    for run in runs:
        bad = np.flatnonzero(~np.isfinite(np.asarray(run["history"][key], dtype=float)))
        if len(bad):
            firsts.append(int(bad[0]) + 1)
    return (min(firsts) if firsts else None), len(firsts)


def diverged_label(method: str, runs: list[dict[str, Any]]) -> tuple[str, int | None]:
    """Full legend text, saying so when some of the method's seeds diverged."""
    first_nan, count = divergence(runs)
    label = LABELS.get(method, method).strip()
    if count:
        label += f" ({count}/{len(runs)} diverged)"
    return label, first_nan


def short_label(method: str, runs: list[dict[str, Any]]) -> tuple[str, int | None]:
    """Compact legend text, saying so when some of the method's seeds diverged."""
    first_nan, count = divergence(runs)
    label = SHORT_LABELS.get(method, method)
    if count:
        label += f" ({count}/{len(runs)} div.)"
    return label, first_nan


# --- figures ------------------------------------------------------------------


def lr_comparison(
    runs_per_method: Runs,
    groups: Groups = METHOD_GROUPS,
    linthresh: float = 1e-4,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Mean selected learning rate per epoch, one panel per family, min-max band over seeds."""
    panels = resolve_groups(runs_per_method, groups)
    if not panels:
        raise ValueError("no runs to plot")
    if figsize is None:
        figsize = column_stack(len(panels)) if len(panels) > 1 else COLUMN_SIZE

    first = runs_per_method[panels[0][1][0]]
    n_epochs = len(first[0]["history"]["lr"])
    epochs = np.arange(1, n_epochs + 1)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            len(panels), 1, figsize=figsize, sharex=True, constrained_layout=True
        )
        axes = np.atleast_1d(axes)

        highest = 0.0
        for ax, (title, methods) in zip(axes, panels, strict=True):
            for method in methods:
                runs = runs_per_method[method]
                mean = np.asarray(mean_history(runs)["lr"], dtype=float)
                lo, hi = band(runs, "lr")
                label, first_nan = short_label(method, runs)
                color = METHOD_COLORS[method]
                ax.plot(
                    epochs,
                    mean,
                    color=color,
                    linestyle=METHOD_STYLES[method],
                    linewidth=1.2,
                    label=label,
                )
                ax.fill_between(epochs, lo, hi, color=color, alpha=0.13, linewidth=0)
                if first_nan is not None and first_nan > 1:
                    # The schedule keeps producing a rate long after the run is dead.
                    ax.plot(
                        first_nan - 1,
                        mean[first_nan - 2],
                        marker="x",
                        markersize=5,
                        markeredgewidth=1.4,
                        color=color,
                        zorder=6,
                    )
                highest = max(highest, float(np.nanmax(hi)))
            ax.grid(True, which="both", alpha=0.25)
            # 7.5pt, not the 6pt of the other figures: this legend carries the
            # divergence counts, which have to survive the print at column width.
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85, handlelength=1.6)
            if title:
                ax.set_title(title, fontsize=7, loc="left", pad=2)

        for ax in axes:
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.set_ylim(0, highest * 1.6 if highest else 1)
            ax.set_ylabel("LR")
        axes[-1].set_xlabel("Epoch")
        axes[-1].set_xlim(0.5, n_epochs + 0.5)
    return fig


def loss_comparison(
    runs_per_method: Runs,
    groups: Groups = METHOD_GROUPS,
    ylim_percentile: tuple[float, float] = (0, 98),
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Validation loss (solid, in the legend) over the training loss (faint).

    Every method's training loss is the batch loss measured where its gradient
    was taken, before the step, so the faint curves are comparable to each
    other. Only the validation curves are named, because ten legend entries per
    panel would drown the panel.
    """
    panels = resolve_groups(runs_per_method, groups)
    if not panels:
        raise ValueError("no runs to plot")
    if figsize is None:
        figsize = column_stack(len(panels)) if len(panels) > 1 else COLUMN_SIZE

    first = runs_per_method[panels[0][1][0]]
    n_epochs = len(first[0]["history"]["val_loss"])
    epochs = np.arange(1, n_epochs + 1)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            len(panels), 1, figsize=figsize, sharex=True, constrained_layout=True
        )
        axes = np.atleast_1d(axes)

        everything = []
        for ax, (title, methods) in zip(axes, panels, strict=True):
            for method in methods:
                runs = runs_per_method[method]
                means = mean_history(runs)
                color, style = METHOD_COLORS[method], METHOD_STYLES[method]

                train = np.asarray(means["train_loss"], dtype=float)
                ax.plot(epochs, train, color=color, linestyle=style, linewidth=0.7, alpha=0.35)

                val = np.asarray(means["val_loss"], dtype=float)
                lo, hi = band(runs, "val_loss")
                label, first_nan = short_label(method, runs)
                ax.plot(epochs, val, color=color, linestyle=style, linewidth=1.2, label=label)
                ax.fill_between(epochs, lo, hi, color=color, alpha=0.12, linewidth=0)
                if first_nan is not None and first_nan > 1:
                    ax.plot(
                        first_nan - 1,
                        val[first_nan - 2],
                        marker="x",
                        markersize=5,
                        markeredgewidth=1.4,
                        color=color,
                        zorder=6,
                    )
                everything.extend([train, val])
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=6, framealpha=0.85, handlelength=1.6)
            if title:
                ax.set_title(title, fontsize=7, loc="left", pad=2)

        joined = np.concatenate(everything)
        finite = joined[np.isfinite(joined)]
        if len(finite):
            low = float(np.percentile(finite, ylim_percentile[0]))
            high = float(np.percentile(finite, ylim_percentile[1]))
            pad = (high - low) * 0.08
            for ax in axes:
                ax.set_ylim(max(0, low - pad), high + pad)
        for ax in axes:
            ax.set_ylabel("Loss")
        axes[-1].set_xlabel("Epoch")
        axes[-1].set_xlim(0.5, n_epochs + 0.5)
    return fig


def polls_per_epoch(
    runs_per_method: Runs,
    methods: Sequence[str] = ("efficient", "efficient_fixed", "efficient_random"),
    max_poll_interval: int = EfficientPolling.max_poll_interval,
) -> Figure:
    """Where each trigger spends its polls, under a budget the three share.

    The ablation's whole argument in one axes: the same number of polls can be
    spread flat, or concentrated where the selection actually changes. The
    dotted floor is one poll per backoff period, ``batches / (K_max + 1)``.
    """
    methods = [m for m in methods if runs_per_method.get(m)]
    if not methods:
        raise ValueError("no Efficient Polling runs to plot")

    record = runs_per_method[methods[0]][0]
    n_epochs = len(record["history"]["polls"])
    epochs = np.arange(1, n_epochs + 1)
    floor = record["batches"] / record["epochs"] / (max_poll_interval + 1)

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=COLUMN_SIZE, constrained_layout=True)

        for method in methods:
            runs = runs_per_method[method]
            mean = np.asarray(mean_history(runs)["polls"], dtype=float)
            lo, hi = band(runs, "polls")
            color = METHOD_COLORS[method]
            ax.plot(
                epochs,
                mean,
                color=color,
                linestyle=METHOD_STYLES[method],
                linewidth=1.2,
                label=SHORT_LABELS[method],
            )
            ax.fill_between(epochs, lo, hi, color=color, alpha=0.12, linewidth=0)

        ax.axhline(floor, color="0.35", linestyle=":", linewidth=1.0)
        ax.annotate(
            f"floor $\\approx$ {floor:.0f}",
            xy=(3, floor),
            fontsize=6,
            color="0.35",
            va="bottom",
            ha="left",
        )

        ax.set_yscale("log")
        ax.set_xlim(0.5, n_epochs + 0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Polls per epoch")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.85, handlelength=1.6)
    return fig


def lr_robustness(runs_by_start: dict[tuple[str, float], list[dict[str, Any]]]) -> Figure:
    """Mean selected learning rate per epoch, one panel per method, one line per starting rate."""
    methods = [
        m for m in ("efficient", "efficient_relative") if any(k[0] == m for k in runs_by_start)
    ]
    if not methods:
        raise ValueError("no initial-rate runs to plot")

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            len(methods),
            1,
            figsize=column_stack(len(methods)),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, method in zip(axes, methods, strict=True):
            for (m, lr0), runs in runs_by_start.items():
                if m != method:
                    continue
                curve = np.asarray(mean_history(runs)["lr"], dtype=float)
                epochs = np.arange(1, len(curve) + 1)
                _, first_nan = short_label(method, runs)
                ax.plot(epochs, curve, lw=1.2, label=f"start {lr0:g}")
                if first_nan is not None:
                    ax.plot(first_nan, curve[first_nan - 1], "x", ms=5, color="k")
            ax.set_yscale("log")
            ax.set_title(LABELS[method].strip())
            ax.set_ylabel("mean LR")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="lower left")
        axes[-1].set_xlabel("Epoch")
    return fig


# --- animations, for the notebook ------------------------------------------------


def animate_lr_overlay(
    runs_per_method: Runs,
    groups: Groups = METHOD_GROUPS,
    interval_ms: int = 80,
    linthresh: float = 1e-4,
) -> animation.FuncAnimation:
    """The learning-rate figure, drawn epoch by epoch, with the same panels and colors.

    In a notebook, show it with ``HTML(animation.to_jshtml())``.
    """
    panels = resolve_groups(runs_per_method, groups)
    curves = {
        method: np.asarray(mean_history(runs_per_method[method])["lr"], dtype=float)
        for _, methods in panels
        for method in methods
    }
    n_epochs = len(next(iter(curves.values())))
    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(9, 3.2 * len(panels)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    artists = []
    for ax, (title, methods) in zip(axes, panels, strict=True):
        highest = max(float(np.nanmax(curves[m])) for m in methods)
        for method in methods:
            color, style = METHOD_COLORS[method], METHOD_STYLES[method]
            label = diverged_label(method, runs_per_method[method])[0]
            (line,) = ax.plot([], [], color=color, linestyle=style, linewidth=1.6, label=label)
            artists.append((method, line, ax.scatter([], [], color=color, s=30, zorder=5)))
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(0, highest * 1.3 if highest else 1)
        ax.set_ylabel("Learning rate")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if title:
            ax.set_title(title, fontsize=10, loc="left")

    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xlim(0.5, n_epochs + 0.5)
    fig.suptitle("Mean learning rate per epoch, all methods")
    marks = [ax.axvline(1, color="gray", ls="--", alpha=0.5) for ax in axes]

    def update(k: int) -> list:
        shown = epochs[: k + 1]
        for method, line, dot in artists:
            values = curves[method][: k + 1]
            line.set_data(shown, values)
            dot.set_offsets(np.c_[shown[-1:], values[-1:]])
        for mark in marks:
            mark.set_xdata([float(epochs[k]), float(epochs[k])])
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_epochs, interval=interval_ms, blit=False)
    plt.close(fig)
    return anim


def animate_loss_overlay(
    runs_per_method: Runs,
    groups: Groups = METHOD_GROUPS,
    interval_ms: int = 80,
    ylim_percentile: tuple[float, float] = (0, 98),
) -> animation.FuncAnimation:
    """The loss figure, drawn epoch by epoch. Solid is validation, faint is training.

    In a notebook, show it with ``HTML(animation.to_jshtml())``.
    """
    panels = resolve_groups(runs_per_method, groups)
    curves = {
        method: mean_history(runs_per_method[method]) for _, methods in panels for method in methods
    }
    n_epochs = len(next(iter(curves.values()))["val_loss"])
    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(9, 3.2 * len(panels)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    artists = []
    for ax, (title, methods) in zip(axes, panels, strict=True):
        for method in methods:
            color, style = METHOD_COLORS[method], METHOD_STYLES[method]
            label = diverged_label(method, runs_per_method[method])[0]
            (faint,) = ax.plot([], [], color=color, linestyle=style, linewidth=0.9, alpha=0.35)
            (solid,) = ax.plot([], [], color=color, linestyle=style, linewidth=1.6, label=label)
            dot = ax.scatter([], [], color=color, s=30, zorder=5)
            artists.append((method, faint, solid, dot))
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if title:
            ax.set_title(title, fontsize=10, loc="left")

    joined = np.concatenate(
        [np.asarray(h[k], dtype=float) for h in curves.values() for k in ("train_loss", "val_loss")]
    )
    finite = joined[np.isfinite(joined)]
    low = float(np.percentile(finite, ylim_percentile[0]))
    high = float(np.percentile(finite, ylim_percentile[1]))
    pad = (high - low) * 0.08
    for ax in axes:
        ax.set_ylim(max(0, low - pad), high + pad)

    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xlim(0.5, n_epochs + 0.5)
    fig.suptitle("Loss per epoch, all methods. Faint curves are the training loss.")
    marks = [ax.axvline(1, color="gray", ls="--", alpha=0.5) for ax in axes]

    def update(k: int) -> list:
        shown = epochs[: k + 1]
        for method, faint, solid, dot in artists:
            train = np.asarray(curves[method]["train_loss"], dtype=float)[: k + 1]
            val = np.asarray(curves[method]["val_loss"], dtype=float)[: k + 1]
            faint.set_data(shown, train)
            solid.set_data(shown, val)
            dot.set_offsets(np.c_[shown[-1:], val[-1:]])
        for mark in marks:
            mark.set_xdata([float(epochs[k]), float(epochs[k])])
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_epochs, interval=interval_ms, blit=False)
    plt.close(fig)
    return anim


# --- writing every figure of a dataset --------------------------------------------


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_figures(results_dir: Path | str, out_dir: Path | str) -> list[Path]:
    """Draw every figure the records of one dataset allow, and return the files written.

    The poll figure needs Efficient Polling runs, and the robustness figure needs
    runs started away from the default rate, which only CIFAR-10 has.
    """
    runs = load_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"no recorded runs in {results_dir}")
    out = Path(out_dir)

    drawings: list[tuple[str, Callable[[], Figure]]] = [
        ("training_comparison_LRs_all", lambda: lr_comparison(runs)),
        ("training_comparison_losses_all", lambda: loss_comparison(runs)),
    ]
    if any(runs.get(m) for m in ("efficient", "efficient_fixed", "efficient_random")):
        drawings.append(("polls_per_epoch", lambda: polls_per_epoch(runs)))
    starts = load_initial_lr_runs(results_dir)
    if len({lr0 for _, lr0 in starts}) > 1:
        drawings.append(("initial_lr_robustness", lambda: lr_robustness(starts)))

    return [_save(draw(), out / f"{name}.png") for name, draw in drawings]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.plots", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--dataset", default="cifar10", choices=list(DATASETS))
    parser.add_argument("--results-dir", default=None, help="default: results/<dataset>")
    parser.add_argument("--out-dir", default=None, help="default: images/<dataset>")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir or REPO_DIR / "results" / args.dataset)
    out_dir = Path(args.out_dir or figures_dir(args.dataset))
    for path in save_figures(results_dir, out_dir):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
