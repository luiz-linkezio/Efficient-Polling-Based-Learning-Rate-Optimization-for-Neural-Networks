"""Efficient Polling: the same selection, polled on demand instead of always."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ._snapshot import StateSnapshot
from .closures import Closure
from .polling import PollingOptimizer, StepInfo, _split_module

__all__ = ["EfficientPollingOptimizer", "EfficientPollingSGD"]


class EfficientPollingOptimizer(PollingOptimizer):
    """Polls only when the selection might have changed.

    The base method's choice is highly redundant: within a training phase,
    consecutive polls keep picking the same learning rate. This variant keeps the
    selection mechanism untouched and makes the *schedule* adaptive:

    1. **Exponential backoff.** After a poll, an unchanged choice doubles the
       poll interval (capped at ``max_poll_interval``); a changed choice resets
       it to 1. Between polls a single blind step reuses the last selected rate.
       Backoff only engages once a poll has shown signal -- at initialization
       every candidate ties on batch accuracy, and reading those ties as
       "stable" would stall training at the smallest candidate.

    2. **Two-tier divergence guard.** Blind steps are unvalidated, so a step at a
       high learning rate can diverge. Both tiers reuse quantities already
       computed:

       * *spike* (prevention): a batch loss above ``spike_factor`` times the loss
         EMA is polled instead of stepped blindly, so the selection criterion can
         reject an explosive step before it lands.
       * *rollback* (recovery): every poll doubles as a known-good checkpoint. A
         non-finite loss, or one above ``rollback_loss``, restores that
         checkpoint and resumes polling every batch.

    On CIFAR-10 this reaches the base method's accuracy while polling ~5% of
    batches, at ~12% over plain SGD instead of +264%.

    Args:
        optimizer: the optimizer whose ``lr`` is polled.
        candidate_lrs: learning rates to poll. See :class:`PollingOptimizer`.
        module: the model, so buffers are restored between trials and by
            rollbacks. See :class:`PollingOptimizer`.
        max_poll_interval: backoff cap ``K_max``, counted in *blind steps between
            polls*, so the steady state is one poll every ``K_max + 1`` batches
            (~11 polls per CIFAR-10 epoch at the default 64). ``0`` polls every
            batch, recovering the base method.
        spike_factor: spike threshold ``gamma`` on the loss EMA. Non-positive
            disables the spike tier.
        loss_ema_beta: EMA decay ``beta`` for the tracked loss.
        rollback_loss: absolute rollback threshold. ``None`` derives it as
            ``rollback_factor`` times the first observed loss, which for a freshly
            initialized classifier is the random-guess loss -- ``2*ln(10)`` for
            CIFAR-10's ten classes, the value used in the paper. Pass a float to
            pin it exactly.
        rollback_factor: multiplier for the derived threshold.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        candidate_lrs: Iterable[float] | None = None,
        module: nn.Module | None = None,
        *,
        max_poll_interval: int = 64,
        spike_factor: float = 3.0,
        loss_ema_beta: float = 0.9,
        rollback_loss: float | None = None,
        rollback_factor: float = 2.0,
    ) -> None:
        super().__init__(optimizer, candidate_lrs=candidate_lrs, module=module)

        if max_poll_interval < 0:
            raise ValueError(f"max_poll_interval must be >= 0, got {max_poll_interval}")
        if not 0.0 <= loss_ema_beta < 1.0:
            raise ValueError(f"loss_ema_beta must be in [0, 1), got {loss_ema_beta}")
        if rollback_loss is not None and not math.isfinite(rollback_loss):
            raise ValueError("rollback_loss must be finite, or None to derive it")

        self.max_poll_interval = int(max_poll_interval)
        self.spike_factor = float(spike_factor)
        self.loss_ema_beta = float(loss_ema_beta)
        self.rollback_factor = float(rollback_factor)

        self._rollback_loss = None if rollback_loss is None else float(rollback_loss)
        self.poll_interval = self._reset_interval()
        self.since_poll = max(1, self.poll_interval)  # so the first batch polls
        self.last_lr: float | None = None
        self.signal_seen = False
        self.ema_loss: float | None = None
        self._checkpoint: StateSnapshot | None = None

    @property
    def rollback_loss(self) -> float | None:
        """Active rollback threshold, ``None`` until derived from the first loss."""
        return self._rollback_loss

    def step(self, closure: Closure) -> StepInfo:
        """Take one step for this batch: poll if needed, otherwise step blind.

        Args:
            closure: re-evaluates the model, returning ``(loss, score)``. See
                :class:`~efficient_polling.closures.Closure`.

        Returns:
            A :class:`StepInfo` describing what happened -- whether this batch was
            polled, spiked or rolled back, and what it cost.
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss, score = closure()
        self._check_differentiable(loss)
        loss_value = float(loss.detach())
        score = float(score)

        # Tier 1: a catastrophic loss means an earlier blind step broke the
        # weights. Restore the last poll's checkpoint and poll again next batch.
        if self._diverged(loss_value):
            if self._checkpoint is not None:
                self._checkpoint.restore()
            self.poll_interval = self._reset_interval()
            self.since_poll = max(1, self.poll_interval)
            return StepInfo(
                lr=self.lr,
                loss=loss_value,
                score=score,
                polled=False,
                rolled_back=True,
                poll_interval=self.poll_interval,
                optimizer_steps=0,
            )

        self._resolve_rollback_loss(loss_value)
        loss.backward()

        # Tier 2: poll a suspicious spike rather than stepping through it.
        spike = (
            self.spike_factor > 0.0
            and self.ema_loss is not None
            and loss_value > self.spike_factor * self.ema_loss
        )

        if spike or self.since_poll >= self.poll_interval:
            info = self._polled_step(closure, loss_value, score, spike)
        else:
            self.optimizer.step()
            self.since_poll += 1
            info = StepInfo(
                lr=self.lr,
                loss=loss_value,
                score=score,
                polled=False,
                poll_interval=self.poll_interval,
                optimizer_steps=1,
            )

        self.ema_loss = (
            loss_value
            if self.ema_loss is None
            else self.loss_ema_beta * self.ema_loss + (1.0 - self.loss_ema_beta) * loss_value
        )
        return info

    # -- internals ---------------------------------------------------------

    def _polled_step(
        self, closure: Closure, loss_value: float, score: float, spike: bool
    ) -> StepInfo:
        poll = self.poll(closure)

        if not self.signal_seen:
            # Nothing to back off from yet: ties everywhere carry no information.
            self.signal_seen = poll.had_signal
            self.poll_interval = self._reset_interval()
        elif poll.lr == self.last_lr:
            self.poll_interval = min(max(self.poll_interval * 2, 1), self.max_poll_interval)
        else:
            self.poll_interval = self._reset_interval()

        self.last_lr = poll.lr
        self.since_poll = 0

        # This point is known good, so it doubles as the rollback checkpoint.
        if self._checkpoint is None:
            self._checkpoint = StateSnapshot(self.optimizer, self.module)
        else:
            self._checkpoint.capture()

        return StepInfo(
            lr=poll.lr,
            loss=loss_value,
            score=score,
            polled=True,
            had_signal=poll.had_signal,
            spike=spike,
            poll_interval=self.poll_interval,
            post_loss=poll.post_loss,
            post_score=poll.post_score,
            optimizer_steps=poll.optimizer_steps,
        )

    def _invalidate_snapshots(self) -> None:
        super()._invalidate_snapshots()
        # The rollback checkpoint is tied to the old parameter list too; keeping it
        # would restore a stale set of tensors on the next divergence.
        self._checkpoint = None

    def _reset_interval(self) -> int:
        """Interval after an unstable poll: every batch, unless K_max forbids it."""
        return min(1, self.max_poll_interval)

    def _diverged(self, loss_value: float) -> bool:
        if not math.isfinite(loss_value):
            return True
        return self._rollback_loss is not None and loss_value > self._rollback_loss

    def _resolve_rollback_loss(self, loss_value: float) -> None:
        if self._rollback_loss is None and math.isfinite(loss_value):
            self._rollback_loss = self.rollback_factor * loss_value

    def _polling_state(self) -> dict[str, Any]:
        state = super()._polling_state()
        state.update(
            poll_interval=self.poll_interval,
            since_poll=self.since_poll,
            last_lr=self.last_lr,
            signal_seen=self.signal_seen,
            ema_loss=self.ema_loss,
            rollback_loss=self._rollback_loss,
        )
        return state

    def _load_polling_state(self, state: dict[str, Any]) -> None:
        super()._load_polling_state(state)
        self.poll_interval = int(state.get("poll_interval", 1))
        self.since_poll = int(state.get("since_poll", 1))
        self.last_lr = state.get("last_lr")
        self.signal_seen = bool(state.get("signal_seen", False))
        self.ema_loss = state.get("ema_loss")
        self._rollback_loss = state.get("rollback_loss")
        # Checkpoint tensors are not serialized: a resumed run has no known-good
        # point until its first poll, which is one batch away at most.
        self._checkpoint = None


class EfficientPollingSGD(EfficientPollingOptimizer):
    """:class:`EfficientPollingOptimizer` over plain SGD -- the configuration in the paper.

    Args:
        params: parameters to optimize, or the model itself (in which case its
            buffers are tracked automatically).
        lr: learning rate before the first poll, and the centre of the default
            candidate set.
        candidate_lrs: see :class:`EfficientPollingOptimizer`.
        module: see :class:`EfficientPollingOptimizer`.

    Remaining keyword arguments configure either :class:`torch.optim.SGD`
    (``momentum``, ``weight_decay``, ``nesterov``, ``dampening``) or the polling
    schedule (``max_poll_interval``, ``spike_factor``, ``loss_ema_beta``,
    ``rollback_loss``, ``rollback_factor``).
    """

    _SGD_KEYS = frozenset({"momentum", "dampening", "weight_decay", "nesterov", "maximize"})

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        *,
        candidate_lrs: Iterable[float] | None = None,
        module: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        params, module = _split_module(params, module)
        sgd_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in self._SGD_KEYS}
        super().__init__(
            torch.optim.SGD(params, lr=lr, **sgd_kwargs),
            candidate_lrs=candidate_lrs,
            module=module,
            **kwargs,
        )
