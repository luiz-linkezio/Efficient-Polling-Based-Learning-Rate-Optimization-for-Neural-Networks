"""Closures and selection criteria fed to the polling optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn

__all__ = ["Closure", "ScoreFn", "accuracy", "negative_loss", "make_closure"]


class Closure(Protocol):
    """Re-evaluates the model at the current parameters.

    A polling step calls the closure once with gradients enabled (to obtain the
    gradient at the polled point) and once per candidate learning rate under
    :func:`torch.no_grad`, to score the resulting trial step.

    The closure must return ``(loss, score)`` and must **not** call
    ``backward()`` or ``zero_grad()`` -- the optimizer owns both. ``score`` is
    maximized: batch accuracy in the paper, but anything comparable works (see
    :func:`negative_loss`).
    """

    def __call__(self) -> tuple[torch.Tensor, float]: ...


ScoreFn = Callable[[torch.Tensor, torch.Tensor], float]


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correct top-1 predictions -- the criterion used in the paper."""
    return (outputs.argmax(dim=1) == targets).float().mean().item()


def negative_loss(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Placeholder score for loss-based selection.

    Returns ``0.0``; :func:`make_closure` special-cases it and scores candidates
    by ``-loss`` instead, which is the natural criterion for regression or any
    task where accuracy is undefined.
    """
    del outputs, targets
    return 0.0


def make_closure(
    model: nn.Module,
    loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    score_fn: ScoreFn = accuracy,
) -> Closure:
    """Build the closure for one batch.

    Args:
        model: module to re-evaluate on every call.
        loss_fn: loss applied to ``(outputs, targets)``.
        inputs: batch inputs, already on the right device.
        targets: batch targets, already on the right device.
        score_fn: candidate score to maximize; defaults to :func:`accuracy`.
            Pass :func:`negative_loss` to select by loss instead.

    Returns:
        A :class:`Closure` suitable for :meth:`~efficient_polling.polling.PollingOptimizer.step`.
    """
    by_loss = score_fn is negative_loss

    def closure() -> tuple[torch.Tensor, float]:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        score = -loss.item() if by_loss else float(score_fn(outputs.detach(), targets))
        return loss, score

    return closure
