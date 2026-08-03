"""Efficient Polling: the same selection, polled on demand instead of always."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ._snapshot import StateSnapshot
from .closures import Closure
from .polling import PollingOptimizer, StepInfo, _split_module

__all__ = ["TRIGGERS", "EfficientPollingOptimizer", "EfficientPollingSGD"]

#: Rules that decide *when* to poll. ``"backoff"`` is the method; the other two
#: exist so an ablation can show that the adaptive trigger is what pays off,
#: rather than the mere fact of polling less often.
TRIGGERS = ("backoff", "fixed", "random")


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

    **Trigger ablation.** ``trigger`` replaces rule 1 without touching rule 2 or
    the selection itself, so the three variants differ only in *when* they poll:

    * ``"backoff"`` -- the method described above.
    * ``"fixed"`` -- poll every ``max_poll_interval + 1`` batches, come what may.
    * ``"random"`` -- poll each batch with probability ``poll_probability``.

    Calibrate the two controls to the polling rate the backoff run actually
    measured (``max_poll_interval = 1/rate - 1``, ``poll_probability = rate``);
    matched rates are what makes the comparison about the trigger rather than
    about how much polling each variant bought.

    Args:
        optimizer: the optimizer whose ``lr`` is polled.
        candidate_lrs: learning rates to poll. See :class:`PollingOptimizer`.
        module: the model, so buffers are restored between trials and by
            rollbacks. See :class:`PollingOptimizer`.
        max_poll_interval: backoff cap ``K_max``, counted in *blind steps between
            polls*, so the steady state is one poll every ``K_max + 1`` batches
            (~11 polls per CIFAR-10 epoch at the default 64). ``0`` polls every
            batch, recovering the base method. Under ``trigger="fixed"`` this is
            not a cap but the interval itself.
        trigger: which rule schedules the polls; one of :data:`TRIGGERS`. The
            ablation controls are not the method -- leave this at ``"backoff"``
            unless you are measuring what the backoff contributes.
        poll_probability: poll rate for ``trigger="random"``, ignored otherwise.
        poll_seed: seeds the private RNG of ``trigger="random"``. Private so a
            run's batch order is the same whichever trigger it uses, which the
            global RNG could not give.
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
        trigger: str = "backoff",
        poll_probability: float = 0.05,
        poll_seed: int = 0,
    ) -> None:
        super().__init__(optimizer, candidate_lrs=candidate_lrs, module=module)

        if max_poll_interval < 0:
            raise ValueError(f"max_poll_interval must be >= 0, got {max_poll_interval}")
        if not 0.0 <= loss_ema_beta < 1.0:
            raise ValueError(f"loss_ema_beta must be in [0, 1), got {loss_ema_beta}")
        if rollback_loss is not None and not math.isfinite(rollback_loss):
            raise ValueError("rollback_loss must be finite, or None to derive it")
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}, got {trigger!r}")
        if not 0.0 < poll_probability <= 1.0:
            raise ValueError(f"poll_probability must be in (0, 1], got {poll_probability}")

        self.max_poll_interval = int(max_poll_interval)
        self.spike_factor = float(spike_factor)
        self.loss_ema_beta = float(loss_ema_beta)
        self.rollback_factor = float(rollback_factor)
        self.trigger = trigger
        self.poll_probability = float(poll_probability)

        self._rollback_loss = None if rollback_loss is None else float(rollback_loss)
        self._rng = random.Random(poll_seed)
        self.poll_interval = self._reset_interval()
        self.since_poll = max(1, self.poll_interval)
        self._force_poll = True  # every trigger polls its first batch
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
                :class:`~efficient_polling_lr_scheduler.closures.Closure`.

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
            self._force_poll = True
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

        if spike or self._should_poll():
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

    def _should_poll(self) -> bool:
        """Whether this batch is polled, spikes aside -- rule 1 and its ablations."""
        if self._force_poll:
            return True
        if self.trigger == "random":
            return self._rng.random() < self.poll_probability
        return self.since_poll >= self.poll_interval

    def _polled_step(
        self, closure: Closure, loss_value: float, score: float, spike: bool
    ) -> StepInfo:
        poll = self.poll(closure)

        if self.trigger != "backoff":
            # The ablation controls hold their schedule regardless of what the
            # poll found; that indifference is the thing being ablated.
            pass
        elif not self.signal_seen:
            # Nothing to back off from yet: ties everywhere carry no information.
            self.signal_seen = poll.had_signal
            self.poll_interval = self._reset_interval()
        elif poll.lr == self.last_lr:
            self.poll_interval = min(max(self.poll_interval * 2, 1), self.max_poll_interval)
        else:
            self.poll_interval = self._reset_interval()

        self.last_lr = poll.lr
        self.since_poll = 0
        self._force_poll = False

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
        """Interval after an unstable poll: every batch, unless K_max forbids it.

        ``trigger="fixed"`` has no unstable state to fall back to -- its interval
        is the constant being tested, so a rollback must not leave it polling
        every batch for the rest of training.
        """
        if self.trigger == "fixed":
            return self.max_poll_interval
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
            force_poll=self._force_poll,
            rng_state=self._rng.getstate(),
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
        self._force_poll = bool(state.get("force_poll", False))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            # Random insists on its exact tuple shape, which a state dict that
            # travelled through JSON no longer has.
            self._rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
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
    ``rollback_loss``, ``rollback_factor``, ``trigger``, ``poll_probability``,
    ``poll_seed``).
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
