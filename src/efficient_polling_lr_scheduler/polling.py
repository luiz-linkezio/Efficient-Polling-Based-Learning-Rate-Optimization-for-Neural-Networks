"""The base polling method: poll every batch (Tan et al.)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ._snapshot import StateSnapshot
from .closures import Closure

__all__ = [
    "PollResult",
    "StepInfo",
    "PollingOptimizer",
    "PollingSGD",
    "default_candidate_lrs",
]

DEFAULT_DECADES: tuple[int, ...] = (-2, -1, 0, 1, 2)


def default_candidate_lrs(lr: float, decades: Sequence[int] = DEFAULT_DECADES) -> tuple[float, ...]:
    """Candidate set spanning ``decades`` orders of magnitude around ``lr``.

    With the defaults and ``lr=1e-3`` this is the paper's
    ``{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}``.
    """
    return tuple(sorted(float(lr) * 10.0**d for d in decades))


@dataclass(frozen=True)
class PollResult:
    """Outcome of evaluating the candidate set on one batch.

    Attributes:
        lr: winning learning rate, already applied to the parameters.
        post_loss: loss after the winning step.
        post_score: score of the winning candidate.
        had_signal: whether the candidates disagreed. All-equal scores mean the
            poll carried no information -- at initialization every candidate
            typically ties on batch accuracy.
        optimizer_steps: optimizer steps the poll cost, ``len(candidates) + 1``.
    """

    lr: float
    post_loss: float
    post_score: float
    had_signal: bool
    optimizer_steps: int


@dataclass(frozen=True)
class StepInfo:
    """What one call to ``step()`` did -- everything worth logging.

    Attributes:
        lr: learning rate actually applied to the parameters.
        loss: loss at the parameters *before* the step.
        score: selection score before the step.
        polled: whether candidates were evaluated on this batch.
        had_signal: whether the candidates disagreed (scores were not all
            equal). Ties everywhere mean the poll carried no information.
        spike: whether a loss spike forced this poll (tier-2 guard).
        rolled_back: whether the step was abandoned and the last known-good
            checkpoint restored (tier-1 guard). No step was taken.
        poll_interval: number of batches until the next scheduled poll.
        post_loss: loss after the applied step, when polling measured it.
        post_score: score after the applied step, when polling measured it.
        optimizer_steps: optimizer steps this batch cost -- ``len(candidates) + 1``
            for a poll, ``1`` for a blind step, ``0`` for a rollback. Sums to the
            cost model in the paper.
    """

    lr: float
    loss: float
    score: float
    polled: bool
    had_signal: bool = False
    spike: bool = False
    rolled_back: bool = False
    poll_interval: int = 1
    post_loss: float | None = None
    post_score: float | None = None
    optimizer_steps: int = 1


class PollingOptimizer:
    """Wraps any optimizer and picks its learning rate by polling every batch.

    At each batch the gradient is computed once at the current parameters, then
    every candidate learning rate is applied as a trial step from an identical
    snapshot and scored by the closure. The winner -- highest score, ties going
    to the smallest learning rate -- is re-applied from that same snapshot.

    This is the replicated base method: accurate, and it costs
    ``len(candidate_lrs) + 1`` optimizer steps per batch. See
    :class:`~efficient_polling_lr_scheduler.efficient.EfficientPollingOptimizer` for the
    variant that polls on demand instead.

    Args:
        optimizer: the optimizer whose ``lr`` is polled. Its update rule is used
            unchanged, so momentum, weight decay or Adam all work.
        candidate_lrs: learning rates to poll. Sorted ascending internally so
            ties always favour the smallest. Defaults to
            :func:`default_candidate_lrs` around the optimizer's initial ``lr``.
        module: the model, so its buffers (BatchNorm running statistics) are
            restored between trials. Required for correctness with any module
            that mutates buffers during the forward pass.

    Note:
        Candidate learning rates are absolute and applied to *every* parameter
        group, which overrides per-group learning rates.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        candidate_lrs: Iterable[float] | None = None,
        module: nn.Module | None = None,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer must be a torch.optim.Optimizer, got {type(optimizer).__name__}"
            )
        if not optimizer.param_groups:
            raise ValueError("optimizer has no parameter groups")

        if candidate_lrs is None:
            candidate_lrs = default_candidate_lrs(float(optimizer.param_groups[0]["lr"]))
        lrs = tuple(sorted(float(lr) for lr in candidate_lrs))
        if not lrs:
            raise ValueError("candidate_lrs must not be empty")
        if lrs[0] < 0.0:
            raise ValueError(f"candidate learning rates must be non-negative, got {lrs[0]}")
        if not all(math.isfinite(lr) for lr in lrs):
            raise ValueError("candidate learning rates must be finite")

        self.optimizer = optimizer
        self.candidate_lrs = lrs
        self.module = module
        self._snapshot: StateSnapshot | None = None

    # -- polling ----------------------------------------------------------

    def poll(self, closure: Closure) -> PollResult:
        """Evaluate every candidate from the current point and apply the winner.

        The gradient must already have been computed at the current parameters;
        every candidate reuses it.
        """
        snapshot = self._snapshot
        if snapshot is None:
            snapshot = self._snapshot = StateSnapshot(self.optimizer, self.module)
        else:
            snapshot.capture()

        best_lr = self.candidate_lrs[0]
        best_score = -math.inf
        best_loss = math.nan
        lowest = math.inf
        highest = -math.inf

        for lr in self.candidate_lrs:  # ascending, so ``>`` keeps the smallest on ties
            snapshot.restore()
            self._apply_lr(lr)
            self.optimizer.step()
            with torch.no_grad():
                loss, score = closure()
            score = float(score)
            lowest = min(lowest, score)
            highest = max(highest, score)
            if score > best_score:
                best_score = score
                best_loss = float(loss)
                best_lr = lr

        snapshot.restore()
        self._apply_lr(best_lr)
        self.optimizer.step()

        return PollResult(
            lr=best_lr,
            post_loss=best_loss,
            post_score=best_score,
            had_signal=highest > lowest,
            optimizer_steps=len(self.candidate_lrs) + 1,
        )

    def step(self, closure: Closure) -> StepInfo:
        """Poll the candidates for this batch and take the winning step.

        Args:
            closure: re-evaluates the model, returning ``(loss, score)``. See
                :class:`~efficient_polling_lr_scheduler.closures.Closure`.

        Returns:
            A :class:`StepInfo` describing the step.
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss, score = closure()
        self._check_differentiable(loss)
        loss.backward()

        poll = self.poll(closure)
        return StepInfo(
            lr=poll.lr,
            loss=float(loss.detach()),
            score=float(score),
            polled=True,
            had_signal=poll.had_signal,
            post_loss=poll.post_loss,
            post_score=poll.post_score,
            optimizer_steps=poll.optimizer_steps,
        )

    # -- optimizer protocol ------------------------------------------------

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @property
    def state(self) -> dict[Any, Any]:
        return self.optimizer.state

    @property
    def lr(self) -> float:
        """Learning rate currently applied to the parameters."""
        return float(self.optimizer.param_groups[0]["lr"])

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self.optimizer.add_param_group(param_group)
        self._invalidate_snapshots()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict(), "polling": self._polling_state()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._load_polling_state(state_dict.get("polling", {}))
        self._invalidate_snapshots()

    def _invalidate_snapshots(self) -> None:
        """Drop every cached snapshot, because the parameter list may have changed.

        Subclasses holding further snapshots must extend this.
        """
        self._snapshot = None

    def _polling_state(self) -> dict[str, Any]:
        return {"candidate_lrs": list(self.candidate_lrs)}

    def _load_polling_state(self, state: dict[str, Any]) -> None:
        if "candidate_lrs" in state:
            self.candidate_lrs = tuple(float(lr) for lr in state["candidate_lrs"])

    # -- internals ---------------------------------------------------------

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @staticmethod
    def _check_differentiable(loss: torch.Tensor) -> None:
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            raise RuntimeError(
                "the closure must return a loss tensor that requires grad; it is "
                "called with gradients enabled and must not detach the loss or "
                "call backward() itself"
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(optimizer={type(self.optimizer).__name__}, "
            f"candidate_lrs={self.candidate_lrs})"
        )


def _split_module(params: Any, module: nn.Module | None) -> tuple[Any, nn.Module | None]:
    """Let the convenience classes take a module where params are expected."""
    if isinstance(params, nn.Module):
        return params.parameters(), module if module is not None else params
    return params, module


class PollingSGD(PollingOptimizer):
    """:class:`PollingOptimizer` over plain SGD -- the base method as published.

    Args:
        params: parameters to optimize, or the model itself (in which case its
            buffers are tracked automatically).
        lr: learning rate before the first poll, and the centre of the default
            candidate set.
        candidate_lrs: see :class:`PollingOptimizer`.
        module: see :class:`PollingOptimizer`.

    Remaining keyword arguments are forwarded to :class:`torch.optim.SGD`.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        *,
        candidate_lrs: Iterable[float] | None = None,
        module: nn.Module | None = None,
        **sgd_kwargs: Any,
    ) -> None:
        params, module = _split_module(params, module)
        super().__init__(
            torch.optim.SGD(params, lr=lr, **sgd_kwargs),
            candidate_lrs=candidate_lrs,
            module=module,
        )
