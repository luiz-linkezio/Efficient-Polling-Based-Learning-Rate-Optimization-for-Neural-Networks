"""Per-step adaptive learning-rate baselines, for comparison against polling.

Neither method polls: they compute a step size from quantities the backward
pass already produced (SPS) or by backtracking until a sufficient-decrease
condition holds (Armijo). They live here so the paper's whole comparison runs
through the same :func:`~efficient_polling_lr_scheduler.training.fit` loop, on
the same measurement convention -- the loss and score reported for a batch are
always the ones at the point where the gradient was taken, *before* the step.

Both reuse :class:`~efficient_polling_lr_scheduler.polling.PollingOptimizer`
for its optimizer protocol and exact snapshot/restore, not for polling: their
candidate set is a formality (the single base learning rate) and
``candidate_lrs`` is never consulted.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ._snapshot import StateSnapshot
from .closures import Closure
from .polling import PollingOptimizer, StepInfo, _split_module

__all__ = ["ArmijoOptimizer", "ArmijoSGD", "SPSOptimizer", "SPSSGD"]


def _grad_norm_sq(optimizer: Optimizer) -> float:
    """Squared L2 norm of the gradient over every parameter the optimizer owns."""
    total = 0.0
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                total += float(param.grad.detach().square().sum())
    return total


class SPSOptimizer(PollingOptimizer):
    """Stochastic Polyak step size: ``lr = loss / ||grad||^2``, capped at ``max_lr``.

    One step per batch, no trial steps -- the cheapest adaptive comparator, and
    the one that shows whether polling's accuracy criterion is doing anything a
    closed-form step size could not.

    Args:
        optimizer: the optimizer whose ``lr`` is overwritten each batch.
        max_lr: cap, needed because ``||grad||^2`` goes to zero near a minimum.
        eps: guards the division.
        module: unused here, accepted for interface symmetry.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float = 10.0,
        eps: float = 1e-8,
        module: nn.Module | None = None,
    ) -> None:
        super().__init__(
            optimizer,
            candidate_lrs=(float(optimizer.param_groups[0]["lr"]),),
            module=module,
        )
        self.max_lr = float(max_lr)
        self.eps = float(eps)

    def step(self, closure: Closure) -> StepInfo:
        self.optimizer.zero_grad(set_to_none=True)
        loss, score = closure()
        self._check_differentiable(loss)
        loss.backward()

        loss_val = float(loss.detach())
        lr = min(self.max_lr, loss_val / (_grad_norm_sq(self.optimizer) + self.eps))
        self._apply_lr(lr)
        self.optimizer.step()

        return StepInfo(
            lr=lr,
            loss=loss_val,
            score=float(score),
            polled=False,
            optimizer_steps=1,
        )


class ArmijoOptimizer(PollingOptimizer):
    """Backtracking line search on the Armijo sufficient-decrease condition.

    Starts at ``lr_max`` and halves until
    ``f(x - lr*g) <= f(x) - alpha*lr*||g||^2`` or ``max_iters`` is exhausted, in
    which case the last (smallest) trial is kept. Every trial is a real
    optimizer step undone by the snapshot, so ``optimizer_steps`` counts them
    and the wall-clock cost is directly comparable to a poll.

    Args:
        optimizer: the optimizer whose ``lr`` the search sets.
        lr_max: first (largest) trial step size.
        alpha: sufficient-decrease constant.
        beta: backtracking factor, applied on every rejected trial.
        max_iters: trial budget per batch.
        module: the model, so buffers are restored between trials.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_max: float = 0.1,
        alpha: float = 1e-4,
        beta: float = 0.5,
        max_iters: int = 10,
        module: nn.Module | None = None,
    ) -> None:
        super().__init__(optimizer, candidate_lrs=(float(lr_max),), module=module)
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must lie in (0, 1), got {beta}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {max_iters}")
        self.lr_max = float(lr_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iters = int(max_iters)

    def step(self, closure: Closure) -> StepInfo:
        self.optimizer.zero_grad(set_to_none=True)
        loss, score = closure()
        self._check_differentiable(loss)
        loss.backward()

        loss_val = float(loss.detach())
        grad_sq = _grad_norm_sq(self.optimizer)

        # Same lazy snapshot as the polling path: every trial departs from an
        # identical pre-step condition.
        snapshot = self._snapshot
        if snapshot is None:
            snapshot = self._snapshot = StateSnapshot(self.optimizer, self.module)
        else:
            snapshot.capture()

        lr = self.lr_max
        post_loss = math.nan
        post_score = math.nan
        trials = 0

        for trial in range(self.max_iters):
            # Recomputed rather than accumulated, so ``lr`` always names the step
            # that is actually on the parameters when the loop exits -- including
            # when the budget runs out and the last trial is kept.
            lr = self.lr_max * self.beta**trial
            snapshot.restore()
            self._apply_lr(lr)
            self.optimizer.step()
            trials += 1
            with torch.no_grad():
                trial_loss, trial_score = closure()
            post_loss = float(trial_loss)
            post_score = float(trial_score)
            if post_loss <= loss_val - self.alpha * lr * grad_sq:
                break

        return StepInfo(
            lr=lr,
            loss=loss_val,
            score=float(score),
            polled=True,  # the batch paid for a search, like a poll
            had_signal=trials > 1,
            post_loss=post_loss,
            post_score=post_score,
            optimizer_steps=trials,
        )


class SPSSGD(SPSOptimizer):
    """:class:`SPSOptimizer` over plain SGD.

    Args:
        params: parameters to optimize, or the model itself.
        lr: initial learning rate, immediately overwritten by the Polyak rule.
        max_lr: see :class:`SPSOptimizer`.
        eps: see :class:`SPSOptimizer`.
        module: see :class:`SPSOptimizer`.

    Remaining keyword arguments are forwarded to :class:`torch.optim.SGD`.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        *,
        max_lr: float = 10.0,
        eps: float = 1e-8,
        module: nn.Module | None = None,
        **sgd_kwargs: Any,
    ) -> None:
        params, module = _split_module(params, module)
        super().__init__(
            torch.optim.SGD(params, lr=lr, **sgd_kwargs),
            max_lr=max_lr,
            eps=eps,
            module=module,
        )


class ArmijoSGD(ArmijoOptimizer):
    """:class:`ArmijoOptimizer` over plain SGD.

    Args:
        params: parameters to optimize, or the model itself.
        lr: initial learning rate; the search sets it every batch.
        lr_max, alpha, beta, max_iters: see :class:`ArmijoOptimizer`.
        module: see :class:`ArmijoOptimizer`.

    Remaining keyword arguments are forwarded to :class:`torch.optim.SGD`.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        *,
        lr_max: float = 0.1,
        alpha: float = 1e-4,
        beta: float = 0.5,
        max_iters: int = 10,
        module: nn.Module | None = None,
        **sgd_kwargs: Any,
    ) -> None:
        params, module = _split_module(params, module)
        super().__init__(
            torch.optim.SGD(params, lr=lr, **sgd_kwargs),
            lr_max=lr_max,
            alpha=alpha,
            beta=beta,
            max_iters=max_iters,
            module=module,
        )
