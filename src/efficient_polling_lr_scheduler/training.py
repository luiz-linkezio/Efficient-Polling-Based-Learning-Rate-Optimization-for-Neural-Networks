"""Optional training helpers -- enough to reproduce the paper in a few lines."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .closures import ScoreFn, accuracy, make_closure
from .polling import PollingOptimizer

__all__ = ["EpochStats", "History", "evaluate", "train_epoch", "fit"]


@dataclass(frozen=True)
class EpochStats:
    """Aggregates one training epoch.

    Attributes:
        loss: mean batch loss, measured at the point where each gradient was
            taken (before the step).
        score: mean batch score, same convention.
        lr: mean applied learning rate.
        polls: batches that evaluated the candidate set.
        rollbacks: batches abandoned by the tier-1 guard.
        spikes: polls forced by the tier-2 guard.
        optimizer_steps: total optimizer steps, for the cost model.
        batches: batches seen.
    """

    loss: float
    score: float
    lr: float
    polls: int = 0
    rollbacks: int = 0
    spikes: int = 0
    optimizer_steps: int = 0
    batches: int = 0


@dataclass
class History:
    """Per-epoch training curves, one list entry per epoch."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    polls: list[int] = field(default_factory=list)
    rollbacks: list[int] = field(default_factory=list)
    spikes: list[int] = field(default_factory=list)
    optimizer_steps: list[int] = field(default_factory=list)
    best_val_acc: float = -math.inf
    best_epoch: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, handy for plotting or serialization."""
        return asdict(self)


def _device_of(model: nn.Module, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str | None = None,
    score_fn: ScoreFn = accuracy,
) -> tuple[float, float]:
    """Mean loss and score over ``loader``, in eval mode.

    Returns:
        ``(loss, score)``.
    """
    target = _device_of(model, device)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    try:
        for inputs, targets in loader:
            inputs = inputs.to(target, non_blocking=True)
            targets = targets.to(target, non_blocking=True)
            outputs = model(inputs)
            losses.append(loss_fn(outputs, targets).item())
            scores.append(float(score_fn(outputs, targets)))
    finally:
        model.train(was_training)
    return _mean(losses), _mean(scores)


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer | PollingOptimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str | None = None,
    score_fn: ScoreFn = accuracy,
) -> EpochStats:
    """Train for one epoch.

    Accepts a polling optimizer or any plain :class:`torch.optim.Optimizer`, so
    the same loop runs the baseline and both polling methods.

    Returns:
        :class:`EpochStats` for the epoch.
    """
    target = _device_of(model, device)
    model.train()
    losses: list[float] = []
    scores: list[float] = []
    lrs: list[float] = []
    polls = rollbacks = spikes = steps = batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(target, non_blocking=True)
        targets = targets.to(target, non_blocking=True)
        batches += 1

        # The same closure serves both paths, so the reported loss and score mean
        # the same thing whichever optimizer is in use.
        closure = make_closure(model, loss_fn, inputs, targets, score_fn)

        if isinstance(optimizer, PollingOptimizer):
            info = optimizer.step(closure)
            steps += info.optimizer_steps
            polls += int(info.polled)
            spikes += int(info.spike)
            if info.rolled_back:
                # The step was abandoned; its metrics describe broken weights.
                rollbacks += 1
                continue
            losses.append(info.loss)
            scores.append(info.score)
            lrs.append(info.lr)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, score = closure()
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            scores.append(score)
            lrs.append(float(optimizer.param_groups[0]["lr"]))

    return EpochStats(
        loss=_mean(losses),
        score=_mean(scores),
        lr=_mean(lrs),
        polls=polls,
        rollbacks=rollbacks,
        spikes=spikes,
        optimizer_steps=steps,
        batches=batches,
    )


def fit(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer | PollingOptimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,
    device: torch.device | str | None = None,
    score_fn: ScoreFn = accuracy,
    checkpoint_path: str | Path | None = None,
    log_fn: Callable[[str], None] | None = print,
    scheduler: Any | None = None,
) -> History:
    """Train for ``epochs``, tracking the best validation score.

    Args:
        model: model to train.
        train_loader: batches of ``(inputs, targets)``.
        val_loader: validation batches.
        optimizer: polling optimizer or plain optimizer.
        loss_fn: training loss.
        epochs: number of epochs.
        device: defaults to the model's device.
        score_fn: metric maximized for checkpointing, and the polling criterion.
        checkpoint_path: where to write ``state_dict()`` whenever the validation
            score improves. ``None`` disables checkpointing.
        log_fn: per-epoch line printer; ``None`` silences it.
        scheduler: optional ``torch.optim.lr_scheduler`` stepped once per epoch,
            after validation. :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
            receives the validation loss; any other scheduler is stepped with no
            argument. Only meaningful for a plain optimizer -- a polling
            optimizer overwrites the learning rate every time it polls.

    Returns:
        The :class:`History` of the run.
    """
    history = History()
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        stats = train_epoch(model, train_loader, optimizer, loss_fn, device, score_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, score_fn)

        history.train_loss.append(stats.loss)
        history.train_acc.append(stats.score)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.lr.append(stats.lr)
        history.polls.append(stats.polls)
        history.rollbacks.append(stats.rollbacks)
        history.spikes.append(stats.spikes)
        history.optimizer_steps.append(stats.optimizer_steps)

        if val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            if path is not None:
                torch.save(model.state_dict(), path)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if log_fn is not None:
            line = (
                f"epoch {epoch:03d}/{epochs} | mean lr {stats.lr:.6f} | "
                f"train loss {stats.loss:.4f} acc {stats.score:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )
            if isinstance(optimizer, PollingOptimizer):
                line += f" | polls {stats.polls}/{stats.batches} rb {stats.rollbacks}"
            log_fn(line)

    if log_fn is not None:
        log_fn(
            f"best val acc {history.best_val_acc:.4f} at epoch {history.best_epoch}"
            + (f" (saved to {path})" if path is not None else "")
        )
    return history
