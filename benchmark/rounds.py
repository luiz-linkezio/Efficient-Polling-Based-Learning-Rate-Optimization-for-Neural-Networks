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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .methods import (
    LABELS,
    METHODS,
    OPTIMIZERS,
    SGD_ONLY,
    Hyperparameters,
    initial_lr,
    optimizer_name,
    rate_text,
)
from .sweep import Experiment, load_runs, summarize

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
    "ceilings_table",
    "rate_table",
    "rate_tables",
    "round_experiment",
    "round_hyperparameters",
    "rounds_table",
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


def rate_table(
    experiment: Experiment,
    rate: float,
    optimizers: Sequence[str] = tuple(OPTIMIZERS),
    methods: Sequence[str] = ROUND_METHODS,
) -> str:
    """Test accuracy of every method started at one rate, as Markdown: one column per optimizer.

    The Initial LR column says what each method was given at that rate, which
    is the same on either optimizer.
    """
    columns = [
        (optimizer_name(optimizer), round_experiment(experiment, Round(optimizer, rate)))
        for optimizer in optimizers
    ]
    at_rate = columns[0][1].hyperparameters
    initial = {method: initial_lr(method, at_rate) for method in methods}
    return _accuracy_table(columns, methods, ROUND_LABELS, len(experiment.seeds), initial)


def rate_tables(
    experiment: Experiment,
    rates: Sequence[float] = RATES,
    methods: Sequence[str] = ROUND_METHODS,
) -> str:
    """One table per starting rate, each headed by the dataset and the rate.

    A rate is what a round asks of a method, so this is the reading that keeps
    the two rounds of a rate together and the three rates apart.
    """
    return "\n\n".join(
        f"### {experiment.spec.name}, starting rate {rate_text(rate)}\n\n"
        + rate_table(experiment, rate, methods=methods)
        for rate in rates
    )


def rounds_table(
    experiment: Experiment,
    rounds: Sequence[Round] = ROUNDS,
    methods: Sequence[str] = ROUND_METHODS,
) -> str:
    """Test accuracy of every method in every round, as Markdown: one column per round.

    The six rounds at a glance; :func:`rate_tables` splits them by starting rate.
    """
    columns = [(round_.title, round_experiment(experiment, round_)) for round_ in rounds]
    return _accuracy_table(columns, methods, ROUND_LABELS, len(experiment.seeds))


def ceilings_table(
    experiment: Experiment,
    ceilings: Sequence[float] = CEILINGS,
    methods: Sequence[str] = CEILING_METHODS,
) -> str:
    """Test accuracy of SPS and Armijo under every ceiling, as Markdown: one column per ceiling."""
    columns = [(f"ceiling {rate_text(c)}", ceiling_experiment(experiment, c)) for c in ceilings]
    labels = {method: LABELS[method].strip() for method in methods}
    return _accuracy_table(columns, methods, labels, len(experiment.seeds))


def _accuracy_table(
    columns: Sequence[tuple[str, Experiment]],
    methods: Sequence[str],
    labels: Mapping[str, str],
    seeds: int,
    initial: Mapping[str, str] | None = None,
) -> str:
    """Mean ± sample deviation of the test accuracy over the seeds recorded so far.

    A cell short of the experiment's seeds says how many it has, and a method a
    column has not run yet is a dash.
    """
    recorded = [load_runs(variant.results_dir) for _, variant in columns]
    extra = [] if initial is None else ["Initial LR"]
    lines = [
        "| " + " | ".join(["Method", *extra, *(title for title, _ in columns)]) + " |",
        "|---|" + "---|" * (len(extra) + len(columns)),
    ]
    for method in methods:
        cells = []
        for runs_per_method in recorded:
            runs = runs_per_method.get(method, [])
            if not runs:
                cells.append("—")
                continue
            mean, deviation = summarize([r["test_acc"] for r in runs])
            cell = f"{mean:.2%} ± {deviation:.2%}"
            if len(runs) < seeds:
                cell += f" ({len(runs)}/{seeds})"
            cells.append(cell)
        row = [labels.get(method, method), *([] if initial is None else [initial[method]]), *cells]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
