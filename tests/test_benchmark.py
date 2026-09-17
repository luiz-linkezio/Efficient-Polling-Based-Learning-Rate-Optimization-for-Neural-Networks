"""What the benchmark builds for a dataset, and how it records and reads runs.

The model and the divergence threshold both depend on the dataset, and both
fail quietly if they don't: a threshold left at CIFAR-10's ten classes sits
*below* the loss a hundred-class run starts at, so every batch would look like
a blow-up and the run would restart forever without erroring once.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmark.__main__ import build_experiment, parse_args
from benchmark.datasets import spec_for
from benchmark.methods import (
    METHODS,
    Ablation,
    Hyperparameters,
    build_epoch_polling,
    build_optimizer,
    run_key,
)
from benchmark.models import build_model
from benchmark.sweep import (
    REPO_DIR,
    Experiment,
    load_initial_lr_runs,
    load_runs,
    results_table,
    summarize,
)


def optimizer_for(method: str, dataset: str, hyperparameters: Hyperparameters | None = None):
    spec = spec_for(dataset)
    return build_optimizer(
        method, build_model(spec), spec, hyperparameters or Hyperparameters(), seed=42
    )


def record(method: str, seed: int, **fields) -> dict:
    """A run record with the fields a table or a figure reads."""
    base = {
        "method": method,
        "seed": seed,
        "epochs": 2,
        "lr0": 1e-3,
        "best_val": 0.5,
        "best_epoch": 2,
        "test_acc": 0.5,
        "test_loss": 1.0,
        "s_per_epoch": 1.0,
        "poll_fraction": 0.0,
        "steps": 20,
        "batches": 20,
        "rollbacks": 0,
        "spikes": 0,
        "history": {"lr": [1e-3, 1e-3], "polls": [0, 0], "val_loss": [1.2, 1.0]},
    }
    return base | fields


# --- models ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "channels", "side", "classes"),
    [
        ("cifar10", 3, 32, 10),
        ("cifar100", 3, 32, 100),
        ("mnist", 1, 28, 10),
        ("fashion_mnist", 1, 28, 10),
    ],
)
def test_the_model_takes_the_shape_the_dataset_has(
    dataset: str, channels: int, side: int, classes: int
) -> None:
    model = build_model(spec_for(dataset))

    logits = model(torch.zeros(2, channels, side, side))

    assert logits.shape == (2, classes)


def test_the_cifar10_model_is_still_the_one_the_paper_counts() -> None:
    """557,898 parameters is quoted in the paper; generalizing the model must
    not have moved it."""
    model = build_model(spec_for("cifar10"))

    assert sum(p.numel() for p in model.parameters()) == 557_898


def test_a_tabular_dataset_gets_a_dense_network_not_a_cnn() -> None:
    """Covertype has no spatial axes at all: the convolutional stack cannot be
    what runs there, and the claim is about the learning rate, not the model."""
    logits = build_model(spec_for("covertype"))(torch.zeros(2, 54))

    assert logits.shape == (2, 7)


# --- methods --------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_every_method_builds(method: str) -> None:
    optimizer, scheduler = optimizer_for(method, "cifar10")

    assert optimizer is not None
    assert (scheduler is not None) == (method in ("cosine", "step", "plateau"))


def test_an_unknown_method_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown method 'sgd'"):
        optimizer_for("sgd", "cifar10")


def test_the_divergence_threshold_follows_the_class_count() -> None:
    optimizer, _ = optimizer_for("efficient", "cifar100")

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(100))


def test_the_relative_variant_gets_the_same_threshold() -> None:
    optimizer, _ = optimizer_for("efficient_relative", "cifar100")

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(100))


def test_the_epoch_controller_gets_the_same_threshold() -> None:
    spec = spec_for("cifar100")
    controller = build_epoch_polling("efficient_relative_epoch", spec, Hyperparameters())

    assert controller.rollback_loss == pytest.approx(2.0 * math.log(100))


def test_cifar10_keeps_the_threshold_the_recorded_runs_used() -> None:
    optimizer, _ = optimizer_for("efficient", "cifar10")

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(10))


def test_the_tabular_threshold_follows_its_seven_classes() -> None:
    optimizer, _ = optimizer_for("efficient", "covertype")

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(7))


def test_polling_keeps_the_papers_grid_around_the_starting_rate() -> None:
    optimizer, _ = optimizer_for("polling", "cifar10")

    assert optimizer.candidate_lrs == pytest.approx((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))


def test_both_ablation_triggers_follow_the_poll_rate() -> None:
    assert Ablation().fixed_interval == 19
    assert Ablation(poll_rate=0.2356).fixed_interval == 3
    assert Ablation(poll_rate=0.2356).poll_probability == 0.2356


# --- where runs live ------------------------------------------------------------


def test_results_land_in_a_directory_of_their_own_per_dataset() -> None:
    assert Experiment("mnist", "/nowhere").results_dir == REPO_DIR / "results" / "mnist"
    assert Experiment("cifar10", "/nowhere").results_dir == REPO_DIR / "results" / "cifar10"


def test_an_explicit_results_dir_still_wins(tmp_path: Path) -> None:
    experiment = Experiment("mnist", "/nowhere", results_dir=tmp_path)

    assert experiment.result_path("efficient", 42) == tmp_path / "efficient_seed42.json"


def test_runs_started_elsewhere_are_recorded_apart_from_the_table() -> None:
    experiment = Experiment("cifar10", "/nowhere")

    assert run_key("efficient", 1e-5) == "efficient_lr1e-05"
    assert experiment.result_path("efficient", 42, lr=0.1).name == "efficient_lr0.1_seed42.json"


def test_checkpoints_of_two_datasets_do_not_overwrite_each_other() -> None:
    mnist = Experiment("mnist", "/nowhere").checkpoint_path("efficient", 42)
    cifar = Experiment("cifar10", "/nowhere").checkpoint_path("efficient", 42)

    assert mnist != cifar
    assert "mnist" in mnist.name


# --- running --------------------------------------------------------------------


def test_a_recorded_run_is_read_back_instead_of_retrained(tmp_path: Path) -> None:
    """The data directory does not exist: touching the data would raise."""
    (tmp_path / "baseline_seed42.json").write_text(json.dumps(record("baseline", 42)))
    experiment = Experiment("cifar10", tmp_path / "missing", results_dir=tmp_path)

    (run,) = experiment.sweep("baseline", seeds=[42])

    assert run["method"] == "baseline"


def test_a_finished_run_is_recorded_whole_and_a_half_written_one_is_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preempted job can die while a record is being written."""
    experiment = Experiment("cifar10", tmp_path / "missing", results_dir=tmp_path)
    monkeypatch.setattr(
        experiment, "run_one", lambda method, seed, epochs, lr: record(method, seed)
    )
    (tmp_path / "cosine_seed42.json.partial").write_text('{"method": "cos')

    experiment.sweep("baseline", seeds=[42])

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "baseline_seed42.json",
        "cosine_seed42.json.partial",
    ]
    assert list(experiment.load_runs()) == ["baseline"]


def test_calibrated_ablations_read_the_efficient_run_of_the_first_seed(tmp_path: Path) -> None:
    efficient = record("efficient", 42, poll_fraction=0.1011)
    (tmp_path / "efficient_seed42.json").write_text(json.dumps(efficient))
    experiment = Experiment(
        "cifar100", "/nowhere", seeds=(42, 43), results_dir=tmp_path, calibrate_ablations=True
    )

    ablation = experiment.hyperparameters_for("efficient_fixed").ablation

    assert ablation.poll_rate == 0.1011
    assert ablation.fixed_interval == 9
    assert experiment.hyperparameters_for("efficient").ablation.poll_rate == 0.05


def test_calibrating_before_the_efficient_run_exists_says_what_is_missing(tmp_path: Path) -> None:
    experiment = Experiment("mnist", "/nowhere", results_dir=tmp_path, calibrate_ablations=True)

    with pytest.raises(FileNotFoundError, match="run efficient on seed 42 first"):
        experiment.hyperparameters_for("efficient_random")


def test_a_run_override_leaves_the_experiment_settings_alone() -> None:
    experiment = Experiment("cifar10", "/nowhere")

    override = experiment.hyperparameters_for("efficient", epochs=3, lr=1e-5)

    assert (override.training.epochs, override.training.lr) == (3, 1e-5)
    assert experiment.hyperparameters.training.epochs == 150
    assert experiment.hyperparameters.training.lr == 1e-3


# --- reading records back ---------------------------------------------------------


def test_records_come_back_in_table_order_without_the_robustness_runs(tmp_path: Path) -> None:
    for method in ("efficient", "baseline", "efficient_lr1e-05", "adam"):
        (tmp_path / f"{method}_seed42.json").write_text(json.dumps(record(method, 42)))

    assert list(load_runs(tmp_path)) == ["baseline", "adam", "efficient"]


def test_runs_recorded_before_the_starting_rate_was_kept_count_as_the_default(
    tmp_path: Path,
) -> None:
    """CIFAR-10's first 55 records have no lr0 field; they started at 1e-3."""
    legacy = record("efficient", 42)
    del legacy["lr0"]
    (tmp_path / "efficient_seed42.json").write_text(json.dumps(legacy))
    (tmp_path / "efficient_lr1e-05_seed42.json").write_text(
        json.dumps(record("efficient_lr1e-05", 42, lr0=1e-5))
    )

    assert list(load_initial_lr_runs(tmp_path)) == [("efficient", 1e-5), ("efficient", 1e-3)]


def test_a_single_run_has_no_spread_and_no_runs_have_no_mean() -> None:
    assert summarize([0.5]) == (0.5, 0.0)
    assert all(math.isnan(v) for v in summarize([]))


def test_the_table_reports_mean_and_sample_deviation() -> None:
    seeds = [record("efficient", 42, test_acc=0.8), record("efficient", 43, test_acc=0.9)]

    row = results_table({"efficient": seeds}).splitlines()[-1]

    assert row.startswith("| Efficient Polling (ours) |")
    assert "85.00% ± 7.07%" in row
    assert row.endswith("| 2 |")


# --- command line -----------------------------------------------------------------


def test_the_command_line_defaults_are_the_recorded_settings() -> None:
    experiment = build_experiment(parse_args(["--data-dir", "/nowhere"]))

    assert experiment.hyperparameters == Hyperparameters()
    assert experiment.spec.key == "cifar10"


def test_a_flag_reaches_the_setting_it_names() -> None:
    args = parse_args(
        ["--data-dir", "/nowhere", "--dataset", "mnist", "--poll-rate", "0.2", "--epochs", "3"]
    )
    experiment = build_experiment(args)

    assert experiment.hyperparameters.ablation.poll_rate == 0.2
    assert experiment.hyperparameters.training.epochs == 3
    assert experiment.results_dir == REPO_DIR / "results" / "mnist"


def test_an_infinite_ceiling_lets_the_relative_window_roam() -> None:
    args = parse_args(["--data-dir", "/nowhere", "--relative-lr-max", "inf"])

    assert build_experiment(args).hyperparameters.relative.lr_max is None
