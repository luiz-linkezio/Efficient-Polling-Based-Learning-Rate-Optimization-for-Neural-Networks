"""The figures are drawn from the records alone, so they are checked against
records written here: no sweep, no dataset, nothing the test did not write."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from benchmark import plots  # noqa: E402
from benchmark.methods import METHODS  # noqa: E402
from benchmark.sweep import REPO_DIR, load_runs  # noqa: E402


def record(method: str, seed: int, epochs: int = 4, batches_per_epoch: int = 213, **fields):
    """A run whose curves have the shape of a real one."""
    history = {
        "train_loss": [2.0 / (e + 1) for e in range(epochs)],
        "train_acc": [0.1 * (e + 1) for e in range(epochs)],
        "val_loss": [2.1 / (e + 1) for e in range(epochs)],
        "val_acc": [0.1 * (e + 1) for e in range(epochs)],
        "lr": [1e-1, 1e-1, 1e-2, 1e-3][:epochs],
        "polls": [40 + seed, 12, 5, 4][:epochs],
        "rollbacks": [0] * epochs,
        "spikes": [0] * epochs,
        "optimizer_steps": [batches_per_epoch] * epochs,
        "best_val_acc": 0.1 * epochs,
    }
    base = {
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "lr0": 1e-3,
        "best_val": 0.4,
        "best_epoch": epochs,
        "test_acc": 0.4,
        "test_loss": 1.0,
        "s_per_epoch": 1.0,
        "poll_fraction": 0.05,
        "steps": batches_per_epoch * epochs,
        "batches": batches_per_epoch * epochs,
        "rollbacks": 0,
        "spikes": 0,
        "history": history,
    }
    return base | fields


def write(results: Path, run: dict) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{run['method']}_seed{run['seed']}.json").write_text(json.dumps(run))


@pytest.fixture
def results(tmp_path: Path) -> Path:
    directory = tmp_path / "results"
    for method in METHODS:
        for seed in (42, 43):
            write(directory, record(method, seed))
    return directory


def test_every_figure_of_a_dataset_is_written(results: Path, tmp_path: Path) -> None:
    written = plots.save_figures(results, tmp_path / "images")

    assert [p.name for p in written] == [
        "training_comparison_LRs_all.png",
        "training_comparison_losses_all.png",
        "polls_per_epoch.png",
    ]
    assert all(p.stat().st_size > 0 for p in written)


def test_the_robustness_figure_needs_runs_started_elsewhere(results: Path, tmp_path: Path) -> None:
    write(results, record("efficient_lr1e-05", 42, lr0=1e-5))
    write(results, record("efficient_relative_lr0.1", 42, lr0=0.1))

    written = plots.save_figures(results, tmp_path / "images")

    assert written[-1].name == "initial_lr_robustness.png"


def test_the_poll_floor_follows_the_datasets_batches_per_epoch(results: Path) -> None:
    """One poll per backoff period: 213 batches over 65 is about 3, not the 11
    that CIFAR-10's 704 batches would draw."""
    figure = plots.polls_per_epoch(load_runs(results))

    (floor,) = [t.get_text() for t in figure.axes[0].texts]

    assert floor == "floor $\\approx$ 3"


def test_the_poll_floor_follows_the_backoff_cap_the_runs_used(results: Path) -> None:
    figure = plots.polls_per_epoch(load_runs(results), max_poll_interval=20)

    (floor,) = [t.get_text() for t in figure.axes[0].texts]

    assert floor == "floor $\\approx$ 10"


def test_a_diverged_seed_is_counted_in_the_legend(tmp_path: Path) -> None:
    directory = tmp_path / "results"
    write(directory, record("cosine", 42))
    diverged = record("cosine", 43)
    diverged["history"]["val_loss"][2:] = [math.nan, math.nan]
    write(directory, diverged)

    figure = plots.lr_comparison(load_runs(directory))

    (label,) = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
    assert label == "cosine (1/2 div.)"


def test_the_baseline_is_named_in_the_legend_after_the_optimizer_and_rate_it_ran_with() -> None:
    """A learning-rate round moves both, so the legend reads them from the record."""
    main_table = plots.short_label("baseline", [record("baseline", 42)])[0]
    adam_round = plots.short_label("baseline", [record("baseline", 42, optimizer="adam", lr0=1e-1)])

    assert (main_table, adam_round[0]) == ("SGD 1e-3", "Adam 1e-1")


def test_figures_of_two_datasets_land_in_different_folders() -> None:
    assert plots.figures_dir("cifar10") == REPO_DIR / "images" / "cifar10"
    assert plots.figures_dir("mnist") != plots.figures_dir("cifar10")


def test_the_command_line_writes_where_it_is_told(results: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"

    plots.main(["--dataset", "covertype", "--results-dir", str(results), "--out-dir", str(out)])

    assert (out / "polls_per_epoch.png").exists()


def test_a_dataset_with_no_records_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no recorded runs"):
        plots.save_figures(tmp_path, tmp_path / "images")


def test_the_animations_build_from_the_records(results: Path) -> None:
    runs = load_runs(results)

    for build in (plots.animate_lr_overlay, plots.animate_loss_overlay):
        assert "<script" in build(runs).to_jshtml()
