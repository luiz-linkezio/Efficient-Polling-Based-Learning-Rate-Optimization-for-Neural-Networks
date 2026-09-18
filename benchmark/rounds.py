"""The learning-rate rounds: every method on one base optimizer, from one rate.

The main table fixes the base optimizer (SGD) and the starting rate (``1e-3``)
and varies the method. A round fixes both at other values, so a table can say
how each method copes with a rate chosen too high or too low, on SGD and then on
Adam, without mixing the optimizer into the comparison: within a round, every
method steps with the same optimizer.

A round gives its rate to each method wherever the method asks for one:

- the baseline holds it for the whole run, and the three schedules start from it
  (the main table starts them at ``1e-1``);
- Polling and Efficient Polling centre their grid on it, two decades either
  side, so from ``1`` the grid reaches ``1e+2``, as in the initial-rate runs;
- Efficient Relative Polling, per batch and per epoch, starts from it, under the
  main table's ``1e-1`` ceiling, raised to the round's rate when the round
  starts above it: the method refuses to start above its ceiling, and every
  other method of that round steps at that rate.

The trigger ablations are calibrated to the poll rate the round's own Efficient
Polling run measured on the first seed.

Three methods sit the rounds out. Adam is the base optimizer of half of them.
SPS and Armijo never read a starting rate: the Polyak step overwrites it on the
first batch, and the line search starts every batch from its ceiling. Their own
test varies that ceiling instead, on SGD only, because both formulas assume the
step follows the gradient.

Records go to ``results/<dataset>/rounds/<optimizer>_lr<rate>/`` and
``results/<dataset>/ceilings/sgd_lr<rate>/``, checkpoints to the same folders
under ``models/``.

The rounds and the ceilings are one study, run as one command, all of it in a
single pool of processes over the GPUs, and read back as one table per
starting rate:

    python -m benchmark.rounds --data-root ~/Datasets
    python -m benchmark.rounds --data-root ~/Datasets --datasets covertype cifar10
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .datasets import DATASETS
from .methods import (
    LABELS,
    METHODS,
    OPTIMIZERS,
    SGD_ONLY,
    Hyperparameters,
    optimizer_name,
    rate_text,
)
from .sweep import TABLE_COLUMNS, Experiment, load_runs, markdown_table, table_cells

if TYPE_CHECKING:
    from .pool import Sweep

__all__ = [
    "CEILINGS",
    "CEILING_METHODS",
    "RATES",
    "ROUNDS",
    "ROUND_LABELS",
    "ROUND_METHODS",
    "Round",
    "ceiling_experiment",
    "ceiling_hyperparameters",
    "DATA_FOLDERS",
    "main",
    "rate_table",
    "rate_tables",
    "round_experiment",
    "round_hyperparameters",
    "study_sweeps",
]

# Very high, middle and very low: the paper's 1e-3, three decades above and four below.
RATES = (1.0, 1e-3, 1e-7)


@dataclass(frozen=True)
class Round:
    """A base optimizer, and the one learning rate every method of the round is given."""

    optimizer: str
    lr: float

    def __post_init__(self) -> None:
        optimizer_name(self.optimizer)
        if not self.lr > 0:
            raise ValueError(f"a round's learning rate must be positive, got {self.lr}")

    @property
    def key(self) -> str:
        """The folder the round records into: ``adam_lr1e-07`` for Adam from ``1e-7``."""
        return f"{self.optimizer}_lr{self.lr:g}"

    @property
    def title(self) -> str:
        """How a table heads the round's column: ``Adam 1e-7``."""
        return f"{optimizer_name(self.optimizer)} {rate_text(self.lr)}"


ROUNDS = tuple(Round(optimizer, lr) for optimizer in OPTIMIZERS for lr in RATES)

# Adam is the base optimizer of half the rounds; SPS and Armijo have a test of their own.
ROUND_METHODS = tuple(method for method in METHODS if method not in ("adam", *SGD_ONLY))
CEILING_METHODS = SGD_ONLY
CEILINGS = RATES

# The summary puts both optimizers side by side, so its rows name neither.
ROUND_LABELS = {
    **{method: LABELS[method].strip() for method in ROUND_METHODS},
    "baseline": "Fixed rate",
    "cosine": "Cosine annealing",
    "step": "Step decay",
    "plateau": "ReduceLROnPlateau",
}


def round_hyperparameters(hyperparameters: Hyperparameters, round_: Round) -> Hyperparameters:
    """``hyperparameters`` on the round's optimizer and rate, the schedules starting from it."""
    training = replace(hyperparameters.training, optimizer=round_.optimizer, lr=round_.lr)
    comparators = replace(hyperparameters.comparators, scheduled_lr=round_.lr)
    relative = hyperparameters.relative
    if relative.lr_max is not None and relative.lr_max < round_.lr:
        relative = replace(relative, lr_max=round_.lr)
    return replace(hyperparameters, training=training, comparators=comparators, relative=relative)


def round_experiment(experiment: Experiment, round_: Round) -> Experiment:
    """The round on ``experiment``'s dataset and seeds, recorded in ``rounds/<key>/``."""
    return experiment.variant(
        f"rounds/{round_.key}",
        round_hyperparameters(experiment.hyperparameters, round_),
        calibrate_ablations=True,
    )


def ceiling_hyperparameters(hyperparameters: Hyperparameters, ceiling: float) -> Hyperparameters:
    """``hyperparameters`` with SPS and Armijo capped at ``ceiling``, on SGD.

    The rate the records keep as ``lr0`` is the ceiling: neither method steps
    with the starting rate it is given.
    """
    training = replace(hyperparameters.training, optimizer="sgd", lr=ceiling)
    comparators = replace(hyperparameters.comparators, sps_max_lr=ceiling, armijo_lr_max=ceiling)
    return replace(hyperparameters, training=training, comparators=comparators)


def ceiling_experiment(experiment: Experiment, ceiling: float) -> Experiment:
    """SPS and Armijo under ``ceiling``, recorded in ``ceilings/sgd_lr<rate>/``."""
    if not ceiling > 0:
        raise ValueError(f"a ceiling must be positive, got {ceiling}")
    return experiment.variant(
        f"ceilings/sgd_lr{ceiling:g}", ceiling_hyperparameters(experiment.hyperparameters, ceiling)
    )


def rate_table(experiment: Experiment, rate: float) -> str:
    """Every run that started from ``rate``, as Markdown, one row per method and optimizer.

    The rows are each round method on SGD, then SPS and Armijo with the rate as
    their ceiling (on SGD, the only optimizer they take), then each round method
    on Adam. A method not run yet has a dash in every column.
    """
    blocks = [
        ("sgd", round_experiment(experiment, Round("sgd", rate)), ROUND_METHODS, ""),
        (
            "sgd",
            ceiling_experiment(experiment, rate),
            CEILING_METHODS,
            f", ceiling {rate_text(rate)}",
        ),
        *(
            (optimizer, round_experiment(experiment, Round(optimizer, rate)), ROUND_METHODS, "")
            for optimizer in OPTIMIZERS
            if optimizer != "sgd"
        ),
    ]
    rows = []
    for optimizer, variant, methods, suffix in blocks:
        runs = load_runs(variant.results_dir)
        for method in methods:
            name = ROUND_LABELS.get(method, LABELS[method].strip()) + suffix
            rows.append([name, optimizer_name(optimizer), *table_cells(runs.get(method, []))])
    return markdown_table(["Method", "Optimizer", *TABLE_COLUMNS], rows)


def rate_tables(experiment: Experiment, rates: Sequence[float] = RATES) -> str:
    """One :func:`rate_table` per starting rate, each headed by the dataset and the rate."""
    return "\n\n".join(
        f"### {experiment.spec.name}, initial LR {rate_text(rate)}\n\n"
        + rate_table(experiment, rate)
        for rate in rates
    )


# Where each dataset sits under a data root: the layout docs/reproducing.md
# describes, and the one the notebook's DATA_DIRS points at.
DATA_FOLDERS = {
    "cifar10": "cifar-10-python/cifar-10-batches-py",
    "cifar100": "cifar-100-python",
    "mnist": "MNIST",
    "fashion_mnist": "fashion-mnist",
    "covertype": "covertype",
}


def study_sweeps(experiment: Experiment) -> list[Sweep]:
    """Every round and ceiling of ``experiment``'s dataset, as sweeps one pool can run."""
    from .pool import Sweep  # the pool imports the command line, which imports this module

    seeds = [str(seed) for seed in experiment.seeds]
    common = ["--dataset", experiment.spec.key, "--data-dir", str(experiment.data_dir)]
    common += ["--seeds", *seeds, "--device", str(experiment.device)]
    sweeps = [
        Sweep(
            round_experiment(experiment, round_),
            [*common, "--round", f"{round_.optimizer}:{round_.lr!r}"],
            ROUND_METHODS,
        )
        for round_ in ROUNDS
    ]
    sweeps += [
        Sweep(
            ceiling_experiment(experiment, ceiling),
            [*common, "--ceiling", repr(ceiling)],
            CEILING_METHODS,
        )
        for ceiling in CEILINGS
    ]
    return sweeps


def main(argv: list[str] | None = None) -> None:
    """Run the whole study, every round and ceiling of every dataset asked for, in one pool."""
    from .pool import RUNS_PER_GPU, run_sweeps, unfinished

    parser = argparse.ArgumentParser(
        prog="python -m benchmark.rounds",
        description="Every learning-rate round and SPS/Armijo ceiling, as one job, "
        "read back as one table per initial LR.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "Datasets",
        help="folder holding the datasets, laid out as "
        + ", ".join(f"{key}: {folder}" for key, folder in DATA_FOLDERS.items())
        + " (default: ~/Datasets)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS), metavar="DATASET"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"runs at a time (default: {RUNS_PER_GPU} per GPU, or 1 without one)",
    )
    parser.add_argument("--device", default=None, help="default: cuda when there is one")
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="print the tables of what is recorded, and run nothing",
    )
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error(f"--workers must be at least 1, got {args.workers}")

    experiments = [
        Experiment(key, args.data_root / DATA_FOLDERS[key], seeds=args.seeds, device=args.device)
        for key in args.datasets
    ]
    message = None
    if not args.tables_only:
        missing = [str(e.data_dir) for e in experiments if not Path(e.data_dir).is_dir()]
        if missing:
            parser.error(f"no dataset at {', '.join(missing)}")
        sweeps = [sweep for experiment in experiments for sweep in study_sweeps(experiment)]
        codes = run_sweeps(sweeps, args.workers)
        message = unfinished(codes, sweeps)

    for experiment in experiments:
        print(f"\n{rate_tables(experiment)}")
    if message:
        sys.exit(message)


if __name__ == "__main__":
    main()
