"""The thirteen configurations the paper compares, and how each one is built.

Every method trains the same network, from the same weights, on the same
batches. What :func:`build_optimizer` returns is the only thing that differs
between two runs of one seed.

A new method is a subclass of ``PollingOptimizer`` in the package, one entry in
:data:`LABELS` and one branch in :func:`build_optimizer`.

Every method except Adam, SPS and Armijo steps with ``Training.optimizer``: SGD
in the main table, Adam as well in the learning-rate rounds of
:mod:`benchmark.rounds`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from efficient_polling_lr_scheduler import (
    SPSSGD,
    ArmijoSGD,
    EfficientPollingOptimizer,
    EfficientRelativeEpochPolling,
    EfficientRelativePollingOptimizer,
    PollingOptimizer,
    default_candidate_lrs,
)

from .datasets import DatasetSpec, blowup_loss

__all__ = [
    "ABLATIONS",
    "LABELS",
    "METHODS",
    "OPTIMIZERS",
    "SGD_ONLY",
    "Ablation",
    "Comparators",
    "EfficientPolling",
    "EfficientRelative",
    "Hyperparameters",
    "Training",
    "base_optimizer",
    "build_epoch_polling",
    "build_optimizer",
    "label",
    "optimizer_name",
    "rate_text",
    "run_key",
]

# In table order, which every table and figure follows: the fixed rate, the
# schedulers, the two rules that measure the step size on the batch, then
# polling and its variants.
LABELS = {
    "baseline": "SGD (fixed 1e-3)",
    "adam": "Adam (1e-3)",
    "cosine": "SGD + cosine annealing",
    "step": "SGD + step decay",
    "plateau": "SGD + ReduceLROnPlateau",
    "sps": "SPS (Polyak)",
    "armijo": "Armijo line search",
    "polling": "Polling (base paper)",
    "efficient": "Efficient Polling (ours)",
    "efficient_fixed": "  ablation: fixed interval",
    "efficient_random": "  ablation: random trigger",
    "efficient_relative": "Efficient Relative Polling (ours, per batch)",
    "efficient_relative_epoch": "Efficient Relative Polling (ours, per epoch)",
}

METHODS = tuple(LABELS)

# The two variants that swap Efficient Polling's trigger and keep everything else.
ABLATIONS = ("efficient_fixed", "efficient_random")

# The optimizers a method can step with, and how the tables write them.
OPTIMIZERS = {"sgd": "SGD", "adam": "Adam"}

# Step-size rules whose formulas assume the step follows the gradient: the Polyak
# step divides by ||g||^2 and the Armijo test asks for a decrease of lr * ||g||^2.
# Both hold for SGD only, so neither runs on another optimizer.
SGD_ONLY = ("sps", "armijo")


@dataclass
class Training:
    """What every method shares."""

    epochs: int = 150
    lr: float = 1e-3  # the baseline's rate, and where every adaptive method starts
    batch_size: int = 64
    val_fraction: float = 0.1
    # Changes the loading speed, never the batches: the samples are already in memory.
    num_workers: int = 0
    # What every method except Adam, SPS and Armijo steps with; see OPTIMIZERS.
    optimizer: str = "sgd"


@dataclass
class Comparators:
    """Settings for the methods the proposal is compared against.

    ``scheduled_lr`` is the *top* of the polling candidate set, not the
    baseline's 1e-3: a decay schedule that starts where the baseline sits has
    nothing to decay from, and comparing against a deliberately crippled
    schedule is how a paper gets accused of picking a straw man.

    ``sps_max_lr`` and ``armijo_lr_max`` are that same ceiling, so no adaptive
    method may take a step the others were never allowed to consider.

    A learning-rate round gives every method the same rate, so there the
    schedules start at the round's rate instead; see :mod:`benchmark.rounds`.
    """

    scheduled_lr: float = 1e-1
    step_size: int = 50  # 1e-1, then 1e-2 from epoch 50 and 1e-3 from epoch 100
    step_gamma: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    cosine_eta_min: float = 0.0
    sps_max_lr: float = 1e-1
    armijo_lr_max: float = 1e-1
    armijo_alpha: float = 1e-4
    armijo_beta: float = 0.5
    armijo_max_iters: int = 10


@dataclass
class EfficientPolling:
    """Efficient Polling. The candidates are the paper's grid, two decades
    either side of the starting rate, and the rollback threshold is twice the
    loss of a uniform guess over the dataset's classes."""

    max_poll_interval: int = 64  # the backoff cap: blind steps between two polls
    spike_factor: float = 3.0  # poll at once when the batch loss exceeds this times its EMA
    loss_ema_beta: float = 0.9


@dataclass
class Ablation:
    """Trigger ablation: same selection, same guard, a different poll schedule.

    The contribution is not polling less, it is deciding *when* to poll. Two
    controls make that testable, a fixed interval and a coin flip, both
    calibrated to the poll rate the backoff variant measures, so the three
    variants poll equally often and differ only in the rule that decides.
    """

    poll_rate: float = 0.05  # the rate the recorded CIFAR-10 ablation used

    @property
    def poll_probability(self) -> float:
        """The random trigger's chance of polling any one batch."""
        return self.poll_rate

    @property
    def fixed_interval(self) -> int:
        """The fixed trigger's blind steps between two polls."""
        return round(1 / self.poll_rate) - 1


@dataclass
class EfficientRelative:
    """Efficient Relative Polling, per batch and per epoch."""

    multiplier: float = 10.0  # {X/m, X, X*m}: one decade apart, like the fixed grid
    # The same 1e-1 ceiling every other method is held to; None lets the window roam.
    lr_max: float | None = 1e-1
    spike_z: float = 3.0  # per batch: deviations above the loss trend that force a poll
    # Per batch: what a poll ranks its trials by, the batch accuracy ("score") or the
    # batch loss ("loss"). The package defaults to the loss since 2.1.0; the recorded
    # runs used the accuracy, so the benchmark keeps it and resumes them unchanged.
    criterion: str = "score"


@dataclass
class Hyperparameters:
    """Every setting of the comparison. The defaults are what the recorded runs used."""

    training: Training = field(default_factory=Training)
    comparators: Comparators = field(default_factory=Comparators)
    efficient: EfficientPolling = field(default_factory=EfficientPolling)
    ablation: Ablation = field(default_factory=Ablation)
    relative: EfficientRelative = field(default_factory=EfficientRelative)


def run_key(method: str, lr: float | None = None) -> str:
    """Name of a run: the method, plus the starting rate when it is not the default.

    The initial-rate robustness runs are keyed apart this way, so they never
    mix with the main table.
    """
    return method if lr is None else f"{method}_lr{lr:g}"


def optimizer_name(optimizer: str) -> str:
    """How the tables write a base optimizer, refusing one the benchmark does not build."""
    try:
        return OPTIMIZERS[optimizer]
    except KeyError:
        known = ", ".join(OPTIMIZERS)
        raise ValueError(f"unknown optimizer {optimizer!r}; known optimizers are {known}") from None


def base_optimizer(optimizer: str, params: Any, lr: float) -> Optimizer:
    """The plain optimizer a method steps with, before polling or a schedule drives it."""
    optimizer_name(optimizer)
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr)


def rate_text(lr: float) -> str:
    """A learning rate as the tables write it: ``1e-3``, not ``0.001``, and ``1``, not ``1e0``."""
    mantissa, exponent = f"{lr:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return mantissa if int(exponent) == 0 else f"{mantissa}e{int(exponent)}"


def initial_lr(method: str, hyperparameters: Hyperparameters) -> str:
    """The learning rate ``method`` starts from under ``hyperparameters``, as a table writes it.

    Not every method starts from ``Training.lr``: the schedules start from
    ``scheduled_lr``, polling chooses from a grid around the rate, SPS and
    Armijo only read a ceiling, and Efficient Relative Polling starts from the
    rate under a ceiling of its own. So a table says, per method, what it was given.
    """
    t, c, r = hyperparameters.training, hyperparameters.comparators, hyperparameters.relative
    lr, scheduled = rate_text(t.lr), rate_text(c.scheduled_lr)
    if method in ("baseline", "adam"):
        return f"{lr}, fixed"
    if method == "cosine":
        return f"{scheduled} → {rate_text(c.cosine_eta_min)}"
    if method == "step":
        drops = max(1, -(-t.epochs // c.step_size))  # one rate per step period the run reaches
        return " → ".join(rate_text(c.scheduled_lr * c.step_gamma**k) for k in range(drops))
    if method == "plateau":
        return f"{scheduled}, ×{c.plateau_factor:g} on plateau"
    if method == "sps":
        return f"Polyak step, up to {rate_text(c.sps_max_lr)}"
    if method == "armijo":
        return f"line search from {rate_text(c.armijo_lr_max)}"
    if method in ("polling", "efficient", *ABLATIONS):
        grid = default_candidate_lrs(t.lr)
        return f"grid {rate_text(grid[0])}–{rate_text(grid[-1])}"
    if method in ("efficient_relative", "efficient_relative_epoch"):
        ceiling = "no ceiling" if r.lr_max is None else f"ceiling {rate_text(r.lr_max)}"
        return f"{lr}, ×÷{r.multiplier:g}, {ceiling}"
    return lr


def label(method: str, optimizer: str = "sgd", lr: float = Training.lr) -> str:
    """How a table names a method that stepped with ``optimizer`` from ``lr``.

    With the defaults this is :data:`LABELS`; a learning-rate round changes the
    optimizer of the fixed rate and of the schedules, and the fixed rate itself.
    """
    if method not in LABELS:
        return method
    name = optimizer_name(optimizer)
    if method == "baseline":
        return f"{name} (fixed {rate_text(lr)})"
    if method == "adam":
        return f"Adam ({rate_text(lr)})"
    return LABELS[method].replace("SGD", name)


def build_optimizer(
    method: str,
    model: nn.Module,
    spec: DatasetSpec,
    hyperparameters: Hyperparameters,
    seed: int,
) -> tuple[Any, Any]:
    """Returns ``(optimizer, scheduler)``; the scheduler is ``None`` for most methods.

    The optimizer is a plain ``torch.optim`` one or one of the package's polling
    optimizers, which wrap it; :func:`~efficient_polling_lr_scheduler.fit` takes both.
    What it steps with is ``Training.optimizer``, except for Adam, which is Adam,
    and for SPS and Armijo, which refuse anything but SGD.
    """
    lr = hyperparameters.training.lr
    name = hyperparameters.training.optimizer
    c = hyperparameters.comparators

    if method in SGD_ONLY and name != "sgd":
        raise ValueError(
            f"{method} is a step-size rule for SGD: its formula assumes the step follows "
            f"the gradient, which {optimizer_name(name)} does not"
        )

    def base(rate: float = lr) -> Optimizer:
        return base_optimizer(name, model.parameters(), rate)

    if method == "baseline":
        return base(), None
    if method == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr), None
    if method == "cosine":
        optimizer = base(c.scheduled_lr)
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hyperparameters.training.epochs, eta_min=c.cosine_eta_min
        )
    if method == "step":
        optimizer = base(c.scheduled_lr)
        return optimizer, torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=c.step_size, gamma=c.step_gamma
        )
    if method == "plateau":
        optimizer = base(c.scheduled_lr)
        return optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=c.plateau_factor, patience=c.plateau_patience
        )
    if method == "sps":
        return SPSSGD(model, lr=lr, max_lr=c.sps_max_lr), None
    if method == "armijo":
        armijo = ArmijoSGD(
            model,
            lr=lr,
            lr_max=c.armijo_lr_max,
            alpha=c.armijo_alpha,
            beta=c.armijo_beta,
            max_iters=c.armijo_max_iters,
        )
        return armijo, None
    if method == "polling":
        return PollingOptimizer(base(), module=model), None
    if method == "efficient_relative":
        r = hyperparameters.relative
        relative = EfficientRelativePollingOptimizer(
            base(),
            module=model,
            multiplier=r.multiplier,
            lr_max=r.lr_max,
            spike_z=r.spike_z,
            rollback_loss=blowup_loss(spec),
            criterion=r.criterion,
        )
        return relative, None
    if method == "efficient_relative_epoch":
        # Per epoch the controller drives the plain optimizer; see build_epoch_polling().
        return base(), None
    if method in ("efficient", *ABLATIONS):
        e = hyperparameters.efficient
        a = hyperparameters.ablation
        # Everything except the trigger is shared, so the ablation cannot be
        # explained by a different guard or a different candidate set.
        kwargs: dict[str, Any] = {
            "spike_factor": e.spike_factor,
            "rollback_loss": blowup_loss(spec),
            "loss_ema_beta": e.loss_ema_beta,
        }
        if method == "efficient":
            kwargs["max_poll_interval"] = e.max_poll_interval
        elif method == "efficient_fixed":
            kwargs.update(trigger="fixed", max_poll_interval=a.fixed_interval)
        else:
            # Seeded per run, so the coin flips differ across seeds the way
            # everything else does, and never touch the global RNG, which would
            # change this run's batch order.
            kwargs.update(trigger="random", poll_probability=a.poll_probability, poll_seed=seed)
        return EfficientPollingOptimizer(base(), module=model, **kwargs), None

    known = ", ".join(METHODS)
    raise ValueError(f"unknown method {method!r}; known methods are {known}")


def build_epoch_polling(
    method: str, spec: DatasetSpec, hyperparameters: Hyperparameters
) -> EfficientRelativeEpochPolling | None:
    """The epoch-level controller for ``efficient_relative_epoch``, ``None`` for the rest."""
    if method != "efficient_relative_epoch":
        return None
    r = hyperparameters.relative
    return EfficientRelativeEpochPolling(
        multiplier=r.multiplier, lr_max=r.lr_max, rollback_loss=blowup_loss(spec)
    )
