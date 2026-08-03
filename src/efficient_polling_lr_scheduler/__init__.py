"""Polling-based learning-rate selection for PyTorch.

Polling picks the learning rate by *measuring* instead of scheduling: at a batch,
every candidate learning rate is applied as a trial step from an identical
snapshot, and the one that most improves batch accuracy is kept. It works
remarkably well and costs several optimizer steps per batch.

**Efficient Polling** keeps that selection but polls on demand -- an exponential
backoff stretches the gap between polls while the choice is stable, and a
two-tier divergence guard protects the unpolled steps. On CIFAR-10 it matches the
base method's accuracy while polling ~5% of batches.

Basic use::

    from efficient_polling_lr_scheduler import EfficientPollingSGD, make_closure

    optimizer = EfficientPollingSGD(model, lr=1e-3)

    for inputs, targets in loader:
        info = optimizer.step(make_closure(model, loss_fn, inputs, targets))
"""

from __future__ import annotations

from ._snapshot import StateSnapshot
from .baselines import SPSSGD, ArmijoOptimizer, ArmijoSGD, SPSOptimizer
from .closures import Closure, ScoreFn, accuracy, make_closure, negative_loss
from .efficient import TRIGGERS, EfficientPollingOptimizer, EfficientPollingSGD
from .polling import (
    PollingOptimizer,
    PollingSGD,
    PollResult,
    StepInfo,
    default_candidate_lrs,
)
from .training import EpochStats, History, evaluate, fit, train_epoch

__version__ = "1.0.1"

__all__ = [
    "ArmijoOptimizer",
    "ArmijoSGD",
    "Closure",
    "EfficientPollingOptimizer",
    "EfficientPollingSGD",
    "EpochStats",
    "History",
    "PollResult",
    "PollingOptimizer",
    "PollingSGD",
    "SPSOptimizer",
    "SPSSGD",
    "ScoreFn",
    "StateSnapshot",
    "StepInfo",
    "TRIGGERS",
    "__version__",
    "accuracy",
    "default_candidate_lrs",
    "evaluate",
    "fit",
    "make_closure",
    "negative_loss",
    "train_epoch",
]
