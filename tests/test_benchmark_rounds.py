"""The learning-rate rounds: every method on one base optimizer, from one rate.

A round moves two settings the main table holds fixed, so what is checked here
is that each method takes the round's optimizer and rate where it asks for them,
that SGD still builds exactly what the recorded runs used, and that no round's
records or checkpoints can land where another's are.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmark.__main__ import build_experiment, main, parse_args
from benchmark.datasets import blowup_loss, spec_for
from benchmark.methods import (
    LABELS,
    METHODS,
    Hyperparameters,
    build_epoch_polling,
    build_optimizer,
    label,
    rate_text,
)
from benchmark.models import build_model
from benchmark.rounds import (
    CEILING_METHODS,
    CEILINGS,
    RATES,
    ROUND_METHODS,
    ROUNDS,
    Round,
    ceiling_experiment,
    ceilings_table,
    rate_tables,
    round_experiment,
    round_hyperparameters,
    rounds_table,
)
from benchmark.sweep import REPO_DIR, Experiment, results_table
from efficient_polling_lr_scheduler import (
    EfficientPollingSGD,
    EfficientRelativePollingSGD,
    PollingSGD,
    fit,
    make_closure,
)


def optimizer_for(method: str, hyperparameters: Hyperparameters, dataset: str = "covertype"):
    spec = spec_for(dataset)
    return build_optimizer(method, build_model(spec), spec, hyperparameters, seed=42)


def in_round(optimizer: str, lr: float) -> Hyperparameters:
    return round_hyperparameters(Hyperparameters(), Round(optimizer, lr))


def record(method: str, seed: int, **fields) -> dict:
    """A run record with the fields a table reads."""
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


def write(path: Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))


# --- what the study is ------------------------------------------------------------


def test_the_rounds_are_both_optimizers_from_three_rates() -> None:
    """The paper's 1e-3, three decades above and four below."""
    assert [(r.optimizer, r.lr) for r in ROUNDS] == [
        ("sgd", 1.0),
        ("sgd", 1e-3),
        ("sgd", 1e-7),
        ("adam", 1.0),
        ("adam", 1e-3),
        ("adam", 1e-7),
    ]


def test_adam_sps_and_armijo_sit_the_rounds_out_and_sps_and_armijo_get_their_own() -> None:
    assert set(METHODS) - set(ROUND_METHODS) == {"adam", "sps", "armijo"}
    assert CEILING_METHODS == ("sps", "armijo")
    assert CEILINGS == RATES


def test_an_unknown_base_optimizer_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown optimizer 'rmsprop'"):
        Round("rmsprop", 1e-3)


# --- how each method takes the round ---------------------------------------------


@pytest.mark.parametrize("round_", ROUNDS, ids=lambda r: r.key)
@pytest.mark.parametrize("method", ROUND_METHODS)
def test_every_round_method_steps_with_the_rounds_optimizer(method: str, round_: Round) -> None:
    built, _ = optimizer_for(method, round_hyperparameters(Hyperparameters(), round_))

    inner = getattr(built, "optimizer", built)  # the polling methods wrap it

    assert type(inner) is {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[round_.optimizer]


@pytest.mark.parametrize("round_", ROUNDS, ids=lambda r: r.key)
@pytest.mark.parametrize("method", ROUND_METHODS)
def test_every_method_of_every_round_gets_through_an_epoch(method: str, round_: Round) -> None:
    """A rate of 1 is further than any recorded run went, and Adam is new to
    every polling method, so each one goes through the real training loop."""
    spec = spec_for("covertype")
    hyperparameters = round_hyperparameters(Hyperparameters(), round_)
    model = build_model(spec)
    optimizer, scheduler = build_optimizer(method, model, spec, hyperparameters, seed=42)
    generator = torch.Generator().manual_seed(7)
    batches = [
        (torch.randn(16, 54, generator=generator), torch.randint(0, 7, (16,), generator=generator))
        for _ in range(3)
    ]

    history = fit(
        model,
        batches,
        batches[:1],
        optimizer,
        nn.CrossEntropyLoss(),
        epochs=1,
        log_fn=None,
        scheduler=scheduler,
        epoch_polling=build_epoch_polling(method, spec, hyperparameters),
    )

    assert len(history.as_dict()["lr"]) == 1


def test_the_baseline_of_an_adam_round_is_adam_at_the_rounds_rate() -> None:
    optimizer, scheduler = optimizer_for("baseline", in_round("adam", 1e-1))

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 1e-1
    assert scheduler is None


@pytest.mark.parametrize("method", ["cosine", "step", "plateau"])
def test_a_round_starts_every_schedule_at_its_rate(method: str) -> None:
    """The main table starts them at 1e-1 whatever the baseline's rate is."""
    optimizer, scheduler = optimizer_for(method, in_round("adam", 1e-5))

    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert scheduler.optimizer is optimizer


def test_a_round_centres_the_polling_grid_on_its_rate() -> None:
    optimizer, _ = optimizer_for("efficient", in_round("adam", 1e-1))

    assert optimizer.candidate_lrs == pytest.approx((1e-3, 1e-2, 1e-1, 1.0, 10.0))


def test_efficient_relative_starts_at_the_rounds_rate_under_the_usual_ceiling() -> None:
    optimizer, _ = optimizer_for("efficient_relative", in_round("sgd", 1e-7))

    assert optimizer.lr == 1e-7
    assert optimizer.lr_max == 1e-1


def test_a_round_that_starts_above_the_relative_ceiling_raises_it_to_its_rate() -> None:
    """Efficient Relative Polling refuses to start above its ceiling, and the
    other methods of that round all step at its rate."""
    hyperparameters = in_round("adam", 1.0)

    optimizer, _ = optimizer_for("efficient_relative", hyperparameters)
    controller = build_epoch_polling(
        "efficient_relative_epoch", spec_for("covertype"), hyperparameters
    )

    assert (optimizer.lr, optimizer.lr_max) == (1.0, 1.0)
    assert controller.lr_max == 1.0


@pytest.mark.parametrize("method", CEILING_METHODS)
def test_sps_and_armijo_refuse_a_base_optimizer_other_than_sgd(method: str) -> None:
    with pytest.raises(ValueError, match="assumes the step follows the gradient"):
        optimizer_for(method, in_round("adam", 1e-3))


def the_recorded_construction(method: str, model: nn.Module, hyperparameters: Hyperparameters):
    """How the recorded runs built the polling methods, before a round could
    change the optimizer: the package's SGD classes."""
    spec = spec_for("covertype")
    lr = hyperparameters.training.lr
    if method == "polling":
        return PollingSGD(model, lr=lr)
    if method == "efficient":
        e = hyperparameters.efficient
        return EfficientPollingSGD(
            model,
            lr=lr,
            spike_factor=e.spike_factor,
            rollback_loss=blowup_loss(spec),
            loss_ema_beta=e.loss_ema_beta,
            max_poll_interval=e.max_poll_interval,
        )
    r = hyperparameters.relative
    return EfficientRelativePollingSGD(
        model,
        lr=lr,
        multiplier=r.multiplier,
        lr_max=r.lr_max,
        spike_z=r.spike_z,
        rollback_loss=blowup_loss(spec),
    )


@pytest.mark.parametrize("method", ["polling", "efficient", "efficient_relative"])
def test_on_sgd_the_methods_take_the_steps_the_recorded_runs_took(method: str) -> None:
    spec = spec_for("covertype")
    hyperparameters = Hyperparameters()
    torch.manual_seed(0)
    model = build_model(spec)
    torch.manual_seed(0)
    recorded_model = build_model(spec)
    optimizer, _ = build_optimizer(method, model, spec, hyperparameters, seed=42)
    recorded = the_recorded_construction(method, recorded_model, hyperparameters)
    loss_fn = nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(7)
    for _ in range(12):
        inputs = torch.randn(16, 54, generator=generator)
        targets = torch.randint(0, 7, (16,), generator=generator)
        optimizer.step(make_closure(model, loss_fn, inputs, targets))
        recorded.step(make_closure(recorded_model, loss_fn, inputs, targets))

    for param, recorded_param in zip(model.parameters(), recorded_model.parameters(), strict=True):
        assert torch.equal(param, recorded_param)


# --- where a round lives ------------------------------------------------------------


def test_each_round_records_and_checkpoints_in_a_folder_of_its_own() -> None:
    experiment = Experiment("mnist", "/nowhere")
    folder = Path("rounds") / "adam_lr0.1"

    adam = round_experiment(experiment, Round("adam", 1e-1))
    sgd = round_experiment(experiment, Round("sgd", 1e-1))

    assert adam.results_dir == REPO_DIR / "results" / "mnist" / folder
    assert adam.checkpoint_path("efficient", 42).parent == REPO_DIR / "models" / folder
    assert adam.checkpoint_path("efficient", 42) != sgd.checkpoint_path("efficient", 42)


def test_the_main_table_never_reads_a_rounds_records(tmp_path: Path) -> None:
    experiment = Experiment("mnist", "/nowhere", results_dir=tmp_path)
    adam = round_experiment(experiment, Round("adam", 1e-5))
    write(experiment.result_path("baseline", 42), record("baseline", 42))
    write(adam.result_path("baseline", 42), record("baseline", 42, optimizer="adam", lr0=1e-5))

    assert [r["lr0"] for r in experiment.load_runs()["baseline"]] == [1e-3]
    assert [r["optimizer"] for r in adam.load_runs()["baseline"]] == ["adam"]


def test_a_round_calibrates_its_ablations_to_its_own_efficient_run(tmp_path: Path) -> None:
    experiment = Experiment("mnist", "/nowhere", results_dir=tmp_path)
    low = round_experiment(experiment, Round("sgd", 1e-5))
    write(low.result_path("efficient", 42), record("efficient", 42, poll_fraction=0.5))

    assert low.hyperparameters_for("efficient_fixed").ablation.fixed_interval == 1
    assert experiment.hyperparameters_for("efficient_fixed").ablation.poll_rate == 0.05


def test_a_round_leaves_the_experiment_alone_and_shares_the_data_it_loaded() -> None:
    experiment = Experiment("mnist", "/nowhere")
    loaded = (torch.zeros(1), torch.ones(1))
    experiment.__dict__["normalization"] = loaded  # as if the cached property had run

    adam = round_experiment(experiment, Round("adam", 1e-1))

    assert adam.normalization is loaded
    assert (adam.hyperparameters.training.optimizer, adam.hyperparameters.training.lr) == (
        "adam",
        1e-1,
    )
    assert experiment.hyperparameters == Hyperparameters()


def test_the_ceiling_test_moves_the_ceiling_of_sps_and_armijo_on_sgd(tmp_path: Path) -> None:
    experiment = Experiment("cifar10", "/nowhere", results_dir=tmp_path)

    ceiling = ceiling_experiment(experiment, 1e-5)
    sps, _ = optimizer_for("sps", ceiling.hyperparameters_for("sps"), "cifar10")
    armijo, _ = optimizer_for("armijo", ceiling.hyperparameters_for("armijo"), "cifar10")

    assert ceiling.results_dir == tmp_path / "ceilings" / "sgd_lr1e-05"
    assert (sps.max_lr, armijo.lr_max) == (1e-5, 1e-5)
    assert isinstance(sps.optimizer, torch.optim.SGD)


# --- how the tables name what ran ---------------------------------------------------


def test_the_main_tables_labels_are_unchanged() -> None:
    assert {method: label(method) for method in METHODS} == LABELS


def test_a_label_names_the_optimizer_and_the_rate_a_run_used() -> None:
    assert label("baseline", "adam", 1e-1) == "Adam (fixed 1e-1)"
    assert label("cosine", "adam", 1e-5) == "Adam + cosine annealing"
    assert label("efficient", "adam", 1e-5) == LABELS["efficient"]


def test_rates_are_written_the_way_the_tables_write_them() -> None:
    rates = (1.0, 1e-1, 1e-3, 1e-7, 2.5e-4, 10.0)

    assert [rate_text(lr) for lr in rates] == ["1", "1e-1", "1e-3", "1e-7", "2.5e-4", "1e1"]


def test_a_rounds_table_names_its_baseline_after_the_record() -> None:
    runs = {"baseline": [record("baseline", 42, optimizer="adam", lr0=0.1)]}

    row = results_table(runs, in_round("adam", 1e-1)).splitlines()[-1]

    assert row.startswith("| Adam (fixed 1e-1) | 1e-1, fixed |")


@pytest.mark.parametrize(
    ("method", "start"),
    [
        ("baseline", "1, fixed"),
        ("cosine", "1 → 0"),
        ("step", "1 → 1e-1 → 1e-2"),
        ("plateau", "1, ×0.5 on plateau"),
        ("polling", "grid 1e-2–1e2"),
        ("efficient_random", "grid 1e-2–1e2"),
        ("efficient_relative", "1, ×÷10, ceiling 1"),
    ],
)
def test_a_rounds_table_says_what_the_round_gave_each_method(method: str, start: str) -> None:
    table = results_table({method: [record(method, 42, lr0=1.0)]}, in_round("sgd", 1.0))

    assert table.splitlines()[-1].split(" | ")[1] == start


def test_the_ceiling_table_says_the_ceiling_sps_and_armijo_were_held_to() -> None:
    hyperparameters = ceiling_experiment(Experiment("mnist", "/nowhere"), 1e-7).hyperparameters
    runs = {m: [record(m, 42, lr0=1e-7)] for m in CEILING_METHODS}

    rows = results_table(runs, hyperparameters).splitlines()[2:]

    assert [row.split(" | ")[1] for row in rows] == [
        "Polyak step, up to 1e-7",
        "line search from 1e-7",
    ]


def test_the_summary_gives_each_round_a_column_and_marks_what_is_missing(tmp_path: Path) -> None:
    experiment = Experiment("mnist", "/nowhere", seeds=(42, 43), results_dir=tmp_path)
    sgd, adam = Round("sgd", 1e-1), Round("adam", 1e-1)
    for seed, acc in ((42, 0.8), (43, 0.9)):
        path = round_experiment(experiment, sgd).result_path("efficient", seed)
        write(path, record("efficient", seed, test_acc=acc))
    path = round_experiment(experiment, adam).result_path("efficient", 42)
    write(path, record("efficient", 42, test_acc=0.5, optimizer="adam"))

    lines = rounds_table(experiment, [sgd, adam], ["baseline", "efficient"]).splitlines()

    assert lines[0] == "| Method | SGD 1e-1 | Adam 1e-1 |"
    assert lines[2] == "| Fixed rate | — | — |"
    assert lines[3] == "| Efficient Polling (ours) | 85.00% ± 7.07% | 50.00% ± 0.00% (1/2) |"


def test_each_starting_rate_gets_a_table_of_its_two_rounds(tmp_path: Path) -> None:
    folder = tmp_path / "rounds"
    write(
        folder / "sgd_lr1" / "baseline_seed42.json",
        record("baseline", 42, lr0=1.0, test_acc=0.7397),
    )
    write(
        folder / "adam_lr1" / "baseline_seed42.json",
        record("baseline", 42, optimizer="adam", lr0=1.0, test_acc=0.5),
    )
    experiment = Experiment("covertype", "/nowhere", seeds=(42,), results_dir=tmp_path)

    tables = rate_tables(experiment, methods=("baseline",))

    assert "### Covertype, starting rate 1\n\n| Method | Initial LR | SGD | Adam |" in tables
    assert "| Fixed rate | 1, fixed | 73.97% ± 0.00% | 50.00% ± 0.00% |" in tables
    assert "| Fixed rate | 1e-7, fixed | — | — |" in tables
    assert "### Covertype, starting rate 1e-3\n" in tables
    assert tables.count("| Fixed rate |") == len(RATES)


def test_the_ceiling_summary_gives_each_ceiling_a_column(tmp_path: Path) -> None:
    experiment = Experiment("cifar10", "/nowhere", seeds=(42,), results_dir=tmp_path)
    path = ceiling_experiment(experiment, 1e-3).result_path("sps", 42)
    write(path, record("sps", 42, test_acc=0.6, lr0=1e-3))

    lines = ceilings_table(experiment, [1e-1, 1e-3]).splitlines()

    assert lines[0] == "| Method | ceiling 1e-1 | ceiling 1e-3 |"
    assert lines[2] == "| SPS (Polyak) | — | 60.00% ± 0.00% |"


# --- command line -------------------------------------------------------------------


def test_a_round_from_the_command_line_is_recorded_as_that_round() -> None:
    args = parse_args(["--data-dir", "/nowhere", "--dataset", "mnist", "--round", "adam:1e-1"])

    experiment = build_experiment(args)

    assert experiment.results_dir == REPO_DIR / "results" / "mnist" / "rounds" / "adam_lr0.1"
    assert experiment.hyperparameters.training.optimizer == "adam"
    assert experiment.hyperparameters.comparators.scheduled_lr == 1e-1
    assert experiment.calibrate_ablations
    assert args.methods == list(ROUND_METHODS)


def test_the_ceiling_test_from_the_command_line_runs_sps_and_armijo() -> None:
    args = parse_args(["--data-dir", "/nowhere", "--ceiling", "1e-5"])

    experiment = build_experiment(args)

    assert experiment.results_dir == REPO_DIR / "results" / "cifar10" / "ceilings" / "sgd_lr1e-05"
    assert experiment.hyperparameters.comparators.armijo_lr_max == 1e-5
    assert args.methods == list(CEILING_METHODS)


@pytest.mark.parametrize(
    ("flags", "complaint"),
    [
        (["--round", "adam"], "OPTIMIZER:RATE"),
        (["--round", "rmsprop:1e-1"], "unknown optimizer"),
        (["--round", "sgd:1e-1", "--methods", "sps"], "not part of a round"),
        (["--ceiling", "1e-1", "--methods", "efficient"], "not part of the ceiling test"),
        (["--round", "sgd:1e-1", "--lr", "1e-5"], "not allowed with"),
    ],
)
def test_a_run_that_does_not_fit_the_study_is_refused(
    flags: list[str], complaint: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--data-dir", "/nowhere", *flags])

    assert complaint in capsys.readouterr().err


def test_a_recorded_round_prints_its_table_without_training(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The data directory does not exist: touching the data would raise."""
    round_dir = tmp_path / "rounds" / "adam_lr0.1"
    write(round_dir / "baseline_seed42.json", record("baseline", 42, optimizer="adam", lr0=0.1))

    main(
        [
            "--data-dir",
            str(tmp_path / "missing"),
            "--results-dir",
            str(tmp_path),
            "--round",
            "adam:1e-1",
            "--methods",
            "baseline",
        ]
    )

    output = capsys.readouterr().out
    assert "round Adam 1e-1" in output
    assert "| Adam (fixed 1e-1) | 1e-1, fixed |" in output
