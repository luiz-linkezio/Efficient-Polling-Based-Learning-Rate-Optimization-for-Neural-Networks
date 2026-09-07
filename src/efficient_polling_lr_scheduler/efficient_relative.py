"""Efficient Relative Polling: candidates around the current rate, polled on a TCP-style schedule.

Where the base method polls a *fixed* candidate set, this variant asks the user
for one learning rate and one multiplier ``m`` and polls three candidates around
whatever rate is currently in use: ``{X/m, X, X*m}``. The winner becomes the new
centre ``X``, so the search window follows the rate wherever training takes it
and nothing has to be known about the right scale in advance. When a poll comes
back blind -- every candidate scoring the same, which batch accuracy does
whenever a single step moves no prediction -- the next poll looks one multiplier
farther in both directions, and keeps widening until it sees a difference; a
poll with signal brings the window back to one multiplier.

Between polls the rate is used blind, and the interval between polls, ``k``, is
the method's confidence: it doubles while polls with signal keep confirming the
choice (a blind poll leaves it unchanged) and has **no fixed cap**. What bounds
it instead is failure, exactly as TCP bounds its congestion window. Every poll
checkpoints the best point seen so far, weights and rate together. When a blind
stretch blows up -- a non-finite loss, or one far above the trend -- training
goes back in time to that best point, ``k`` drops to zero, and the *ceiling* for
the next slow start becomes half the interval that blew up. Growth is
exponential up to that ceiling and linear above it, so the work discarded by a
blow-up halves every time one happens.

Two tie-break rules keep the relative window from drifting on its own. A poll
whose candidates all score the same carries no information, so an ordinary poll
keeps the current rate on a tie. The one poll right after a restart breaks ties
*downward* instead, because a restart has a single cause -- the rate was too
high for blind steps -- and that is the only way the rate descends once the
batch accuracy saturates and the criterion goes blind.

The trend of the loss is tracked the way Adam tracks gradients: a
bias-corrected exponential mean and a second moment of the deviations, so a
spike is measured in standard deviations above the trend rather than as a fixed
ratio to it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from torch import nn
from torch.optim import Optimizer

from ._snapshot import StateSnapshot
from .closures import Closure
from .polling import _SGD_KEYS, PollingOptimizer, StepInfo, _split_module
from .training import EpochStats

__all__ = [
    "Backoff",
    "Window",
    "Trend",
    "EfficientRelativePollingOptimizer",
    "EfficientRelativePollingSGD",
    "EfficientRelativeEpochPolling",
]


class Backoff:
    """The poll interval as a TCP congestion window.

    ``interval`` (``k``) is the number of blind steps between two polls. It
    starts at zero -- poll every step -- and evolves on three events:

    * :meth:`polled` with signal and an unchanged choice doubles it (slow
      start), with no cap until a restart has set a ceiling; above the ceiling
      it grows by one per such poll (congestion avoidance). A changed choice
      zeroes it. A blind poll -- every candidate scoring the same -- leaves it
      as it is: that is not evidence of stability, so an initialization plateau
      keeps polling at whatever interval it had, which is zero, while the
      window widens until a poll can see.
    * :meth:`restart` -- a blow-up -- zeroes it and sets the ceiling to half
      the interval that blew up, the way TCP halves its threshold on a loss.
    * :meth:`blind_step` counts down to the next poll.

    ``after_restart`` marks the first poll after a restart, which the optimizer
    breaks ties on differently. That poll never grows the interval: it only
    re-measures the rate at the restored point.
    """

    def __init__(self) -> None:
        self.interval = 0
        self.ceiling: int | None = None
        self.since_poll = 0
        self.signal_seen = False
        self.after_restart = False

    def due(self) -> bool:
        """Whether the current step should poll."""
        return self.since_poll >= self.interval

    def blind_step(self) -> None:
        self.since_poll += 1

    def polled(self, changed: bool, had_signal: bool, scheduled: bool = True) -> None:
        """Record a poll's outcome and set the interval until the next one.

        Args:
            changed: the poll moved the rate.
            had_signal: the candidates did not all score the same. A blind
                poll is evidence of nothing, so it leaves the interval as it
                is; the window widens for the next one instead.
            scheduled: the poll was due by the interval, rather than forced by
                a spike. Only scheduled polls grow the interval -- once per
                interval, as TCP grows its window once per round trip -- since
                spikes come in bursts and each would otherwise double it.
        """
        self.since_poll = 0
        self.signal_seen = self.signal_seen or had_signal
        if self.after_restart:
            self.after_restart = False
            self.interval = 0
        elif changed:
            self.interval = 0
        elif had_signal and scheduled:
            self.interval = self._grow(self.interval)

    def restart(self) -> None:
        """A blow-up: poll now, and remember how far the last slow start got."""
        self.ceiling = max(1, self.interval // 2)
        self.interval = 0
        self.since_poll = 0
        self.after_restart = True

    def _grow(self, interval: int) -> int:
        doubled = max(1, 2 * interval)
        if self.ceiling is not None and doubled > self.ceiling:
            return max(interval + 1, self.ceiling)
        return doubled

    def state_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "ceiling": self.ceiling,
            "since_poll": self.since_poll,
            "signal_seen": self.signal_seen,
            "after_restart": self.after_restart,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.interval = int(state["interval"])
        ceiling = state.get("ceiling")
        self.ceiling = None if ceiling is None else int(ceiling)
        self.since_poll = int(state["since_poll"])
        self.signal_seen = bool(state["signal_seen"])
        self.after_restart = bool(state["after_restart"])


class Trend:
    """Adam's first and second moments, applied to a scalar such as the loss.

    ``mean`` is a bias-corrected exponential moving average, so it reads as the
    first sample after one update instead of lagging from zero. ``std`` is the
    square root of a bias-corrected moving average of the squared deviation of
    each sample from the mean before it, which is the spread the samples
    actually show around the trend.

    Args:
        beta_mean: decay of the mean, Adam's ``beta1``.
        beta_var: decay of the deviation moment, Adam's ``beta2``. Slower, as in
            Adam, because a spread needs more samples than a level does.
        warmup: samples the moments need before :meth:`is_spike` trusts them.
            A spread estimated from a handful of samples is noise, and a
            spike test on it fires on every wobble.
    """

    def __init__(self, beta_mean: float = 0.9, beta_var: float = 0.999, warmup: int = 50) -> None:
        for name, beta in (("beta_mean", beta_mean), ("beta_var", beta_var)):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {beta}")
        self.beta_mean = float(beta_mean)
        self.beta_var = float(beta_var)
        self.warmup = int(warmup)
        self._m = 0.0
        self._v = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        deviation = 0.0 if self.count == 0 else value - self.mean
        self._m = self.beta_mean * self._m + (1.0 - self.beta_mean) * value
        self._v = self.beta_var * self._v + (1.0 - self.beta_var) * deviation * deviation
        self.count += 1

    @property
    def mean(self) -> float:
        if self.count == 0:
            return math.nan
        return self._m / (1.0 - self.beta_mean**self.count)

    @property
    def std(self) -> float:
        if self.count == 0:
            return math.nan
        return math.sqrt(max(self._v / (1.0 - self.beta_var**self.count), 0.0))

    def is_spike(self, value: float, z: float | None) -> bool:
        """Whether ``value`` sits more than ``z`` deviations above the trend.

        ``None`` disables the test; so does a trend that has not warmed up.
        """
        if z is None or self.count < self.warmup:
            return False
        # A spread of exactly zero would make rounding noise a spike.
        spread = max(self.std, 1e-9 * abs(self.mean))
        return value > self.mean + z * spread

    def state_dict(self) -> dict[str, Any]:
        return {"m": self._m, "v": self._v, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._m = float(state["m"])
        self._v = float(state["v"])
        self.count = int(state["count"])


def _check_bounds(lr: float, lr_min: float | None, lr_max: float | None) -> None:
    for name, bound in (("lr_min", lr_min), ("lr_max", lr_max)):
        if bound is not None and not (math.isfinite(bound) and bound > 0.0):
            raise ValueError(f"{name} must be a positive finite number, got {bound}")
    if lr_min is not None and lr_max is not None and lr_min > lr_max:
        raise ValueError(f"lr_min ({lr_min}) must not exceed lr_max ({lr_max})")
    if (lr_min is not None and lr < lr_min) or (lr_max is not None and lr > lr_max):
        raise ValueError(f"the initial lr {lr} lies outside [{lr_min}, {lr_max}]")


class Window:
    """The three candidates around the current rate, and how far out they reach.

    A poll tries ``{X/f, X, X*f}`` with ``f = multiplier ** reach``. ``reach``
    starts at one and grows by one after every *blind* poll -- all candidates
    scoring the same -- so a criterion that cannot tell one multiplier apart
    is asked about two, then three, until it sees a difference; a poll with
    signal brings the reach back to one. Batch accuracy is blind whenever a
    single step moves no prediction, which at initialization covers every rate
    up to about a decade below the one that first learns; without the widening
    a window started there would never find it.

    Args:
        multiplier: spacing ``m`` between the centre and its neighbours.
        lr_min: optional floor the candidates fold into.
        lr_max: optional ceiling the candidates fold into.
        max_reach: cap on ``reach``, so a blind stretch cannot widen the
            window without limit.
    """

    def __init__(
        self,
        multiplier: float = 10.0,
        lr_min: float | None = None,
        lr_max: float | None = None,
        max_reach: int = 6,
    ) -> None:
        if not (math.isfinite(multiplier) and multiplier > 1.0):
            raise ValueError(f"multiplier must be greater than 1, got {multiplier}")
        for name, bound in (("lr_min", lr_min), ("lr_max", lr_max)):
            if bound is not None and not (math.isfinite(bound) and bound > 0.0):
                raise ValueError(f"{name} must be a positive finite number, got {bound}")
        if lr_min is not None and lr_max is not None and lr_min > lr_max:
            raise ValueError(f"lr_min ({lr_min}) must not exceed lr_max ({lr_max})")
        if max_reach < 1:
            raise ValueError(f"max_reach must be at least 1, got {max_reach}")
        self.multiplier = float(multiplier)
        self.lr_min = None if lr_min is None else float(lr_min)
        self.lr_max = None if lr_max is None else float(lr_max)
        self.max_reach = int(max_reach)
        self.reach = 1

    def candidates(self, centre: float, ascending: bool) -> tuple[float, ...]:
        """The candidates in tie-break order, folded into the bounds.

        An ordinary poll lists the centre first so ties keep it; the poll after
        a restart lists the rates ascending so ties go one notch down. Two
        candidates folded into the same bound leave one.
        """
        factor = self.multiplier**self.reach
        lower, upper = centre / factor, centre * factor
        order = (lower, centre, upper) if ascending else (centre, lower, upper)
        candidates: list[float] = []
        for lr in order:
            if self.lr_min is not None:
                lr = max(lr, self.lr_min)
            if self.lr_max is not None:
                lr = min(lr, self.lr_max)
            if lr not in candidates:
                candidates.append(lr)
        return tuple(candidates)

    def polled(self, had_signal: bool) -> None:
        self.reach = 1 if had_signal else min(self.reach + 1, self.max_reach)

    def reset(self) -> None:
        self.reach = 1

    def state_dict(self) -> dict[str, Any]:
        return {"reach": self.reach}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.reach = int(state["reach"])


def _check_z(name: str, z: float | None) -> float | None:
    if z is None:
        return None
    if not (math.isfinite(z) and z > 0.0):
        raise ValueError(f"{name} must be a positive number or None, got {z}")
    return float(z)


class EfficientRelativePollingOptimizer(PollingOptimizer):
    """Polls ``{X/m, X, X*m}`` around the current rate, on a TCP-style schedule.

    See the module docstring for the method. Compared with
    :class:`~efficient_polling_lr_scheduler.efficient.EfficientPollingOptimizer`:

    * the candidates are *relative* -- three rates one ``multiplier`` apart,
      centred on the rate in use, and the winner becomes the new centre; a
      poll that comes back blind widens the window by another multiplier for
      the next one (:class:`Window`);
    * the poll interval has no fixed cap: it doubles on every scheduled poll
      with signal that keeps the rate, and only a blow-up bounds it, by
      restoring the best point and halving the ceiling of the next slow start
      (:class:`Backoff`);
    * the rollback checkpoint is the *best* point seen, not the last poll --
      weights, optimizer state, buffers and the rate that led there -- and
      going back to it also rewinds the loss trend, so the restart really is a
      return in time;
    * spikes are measured in deviations above an Adam-style trend
      (:class:`Trend`) rather than as a fixed ratio to a plain EMA.

    Ties on an ordinary poll keep the current rate. The poll right after a
    restart breaks ties downward, one notch: a restart's only cause is a rate
    too high for blind steps, and that notch is how the rate descends once the
    batch accuracy saturates and the criterion can no longer tell candidates
    apart.

    Args:
        optimizer: the optimizer whose ``lr`` is polled; its initial ``lr`` is
            the only learning rate the user chooses.
        module: the model, so buffers are restored between trials and by
            restarts. See :class:`PollingOptimizer`.
        multiplier: spacing ``m`` between neighbouring candidates. ``10`` is the
            decade spacing of the paper's fixed grid.
        lr_min: optional floor for the candidates; ``None`` leaves the window free.
        lr_max: optional ceiling. The paper's comparison caps every method at
            the top of the fixed grid, ``1e-1``, so no method may take a step
            the others were never allowed to consider.
        spike_z: deviations above the trend that force a poll instead of a
            blind step. ``None`` disables the spike tier.
        blowup_z: deviations above the trend that count as a blow-up and
            trigger a restart, on top of the absolute ``rollback_loss`` test.
            ``None``, the default, leaves blow-ups to non-finite losses and the
            absolute threshold.
        trend_betas: ``(beta_mean, beta_var)`` of the fast trend the spike and
            blow-up tests read, Adam's ``(0.9, 0.999)``.
        best_beta: decay of the slow trend that decides whether a polled point
            is the best so far. Slower than the spike trend, so the best point
            is not chosen on batch noise.
        warmup: samples the trends need before the deviation tests apply.
        max_reach: how many multipliers out the window may widen while polls
            come back blind; see :class:`Window`.
        rollback_loss: absolute blow-up threshold. ``None`` derives it as
            ``rollback_factor`` times the first observed loss.
        rollback_factor: multiplier for the derived threshold.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        module: nn.Module | None = None,
        *,
        multiplier: float = 10.0,
        lr_min: float | None = None,
        lr_max: float | None = None,
        spike_z: float | None = 3.0,
        blowup_z: float | None = None,
        trend_betas: tuple[float, float] = (0.9, 0.999),
        best_beta: float = 0.99,
        warmup: int = 50,
        max_reach: int = 6,
        rollback_loss: float | None = None,
        rollback_factor: float = 2.0,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer must be a torch.optim.Optimizer, got {type(optimizer).__name__}"
            )
        if not optimizer.param_groups:
            raise ValueError("optimizer has no parameter groups")
        lr = float(optimizer.param_groups[0]["lr"])
        # The fixed candidate set is a formality here: every poll builds its own.
        super().__init__(optimizer, candidate_lrs=(lr,), module=module)

        self.window = Window(multiplier, lr_min, lr_max, max_reach)
        _check_bounds(lr, self.window.lr_min, self.window.lr_max)
        if rollback_loss is not None and not math.isfinite(rollback_loss):
            raise ValueError("rollback_loss must be finite, or None to derive it")

        self.spike_z = _check_z("spike_z", spike_z)
        self.blowup_z = _check_z("blowup_z", blowup_z)
        self.rollback_factor = float(rollback_factor)
        self._rollback_loss = None if rollback_loss is None else float(rollback_loss)

        beta_mean, beta_var = trend_betas
        self.backoff = Backoff()
        self.trend = Trend(beta_mean, beta_var, warmup)
        self.slow_trend = Trend(best_beta, beta_var, warmup)

        self.best_lr: float | None = None
        self.best_trend = math.inf
        self._best: StateSnapshot | None = None
        self._best_trends: tuple[dict[str, Any], dict[str, Any]] | None = None

    @property
    def rollback_loss(self) -> float | None:
        """Active absolute blow-up threshold, ``None`` until derived from the first loss."""
        return self._rollback_loss

    @property
    def multiplier(self) -> float:
        return self.window.multiplier

    @property
    def lr_min(self) -> float | None:
        return self.window.lr_min

    @property
    def lr_max(self) -> float | None:
        return self.window.lr_max

    def step(self, closure: Closure) -> StepInfo:
        """Take one step for this batch: poll if due or spiking, otherwise step blind.

        A blow-up takes no step at all: it restores the best point and returns
        a :class:`StepInfo` with ``rolled_back`` set, and the next batch polls.
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss, score = closure()
        self._check_differentiable(loss)
        loss_value = float(loss.detach())
        score = float(score)

        if self._blew_up(loss_value):
            self._restore_best()
            self.backoff.restart()
            self.window.reset()
            return StepInfo(
                lr=self.lr,
                loss=loss_value,
                score=score,
                polled=False,
                rolled_back=True,
                poll_interval=self.backoff.interval,
                optimizer_steps=0,
            )

        self._resolve_rollback_loss(loss_value)
        loss.backward()

        # The spike test reads the trend *before* this batch joins it.
        spike = self.trend.is_spike(loss_value, self.spike_z)
        self.trend.update(loss_value)
        self.slow_trend.update(loss_value)

        due = self.backoff.due()
        if spike or due:
            return self._polled_step(closure, loss_value, score, spike, scheduled=due)

        self.optimizer.step()
        self.backoff.blind_step()
        return StepInfo(
            lr=self.lr,
            loss=loss_value,
            score=score,
            polled=False,
            poll_interval=self.backoff.interval,
            optimizer_steps=1,
        )

    # -- internals ---------------------------------------------------------

    def _polled_step(
        self, closure: Closure, loss_value: float, score: float, spike: bool, scheduled: bool
    ) -> StepInfo:
        centre = self.lr
        # The pre-step point is what the slow trend has been measuring; if it is
        # the lowest yet, this is the point a blow-up will return to.
        if self._improves(self.slow_trend.mean):
            self._capture_best(centre)

        candidates = self.window.candidates(centre, ascending=self.backoff.after_restart)
        poll = self.poll(closure, candidates)
        self.backoff.polled(
            changed=poll.lr != centre, had_signal=poll.had_signal, scheduled=scheduled
        )
        self.window.polled(poll.had_signal)

        return StepInfo(
            lr=poll.lr,
            loss=loss_value,
            score=score,
            polled=True,
            had_signal=poll.had_signal,
            spike=spike,
            poll_interval=self.backoff.interval,
            post_loss=poll.post_loss,
            post_score=poll.post_score,
            optimizer_steps=poll.optimizer_steps,
        )

    def _blew_up(self, loss_value: float) -> bool:
        if not math.isfinite(loss_value):
            return True
        if self._rollback_loss is not None and loss_value > self._rollback_loss:
            return True
        return self.trend.is_spike(loss_value, self.blowup_z)

    def _resolve_rollback_loss(self, loss_value: float) -> None:
        if self._rollback_loss is None and math.isfinite(loss_value):
            self._rollback_loss = self.rollback_factor * loss_value

    def _improves(self, level: float) -> bool:
        """Whether ``level`` beats the best trend by more than rounding noise.

        A bias-corrected mean of a constant series wanders by ~1e-16 from step
        to step; read literally, that would move the best point forward on a
        loss that has not improved at all.
        """
        if not math.isfinite(self.best_trend):
            return math.isfinite(level)
        return level < self.best_trend - 1e-6 * max(1.0, abs(self.best_trend))

    def _capture_best(self, lr: float) -> None:
        if self._best is None:
            self._best = StateSnapshot(self.optimizer, self.module)
        else:
            self._best.capture()
        self.best_lr = lr
        self.best_trend = self.slow_trend.mean
        self._best_trends = (self.trend.state_dict(), self.slow_trend.state_dict())

    def _restore_best(self) -> None:
        """Back in time: weights, optimizer state, buffers, rate and trend."""
        if self._best is None or self.best_lr is None or self._best_trends is None:
            return  # nothing known-good yet: the next batch polls from here
        self._best.restore()
        self._apply_lr(self.best_lr)
        fast, slow = self._best_trends
        self.trend.load_state_dict(fast)
        self.slow_trend.load_state_dict(slow)

    def _invalidate_snapshots(self) -> None:
        super()._invalidate_snapshots()
        self._best = None

    def _polling_state(self) -> dict[str, Any]:
        state = super()._polling_state()
        state.update(
            lr=self.lr,
            backoff=self.backoff.state_dict(),
            window=self.window.state_dict(),
            trend=self.trend.state_dict(),
            slow_trend=self.slow_trend.state_dict(),
            best_lr=self.best_lr,
            rollback_loss=self._rollback_loss,
        )
        return state

    def _load_polling_state(self, state: dict[str, Any]) -> None:
        super()._load_polling_state(state)
        if "lr" in state:
            self._apply_lr(float(state["lr"]))
        if "backoff" in state:
            self.backoff.load_state_dict(state["backoff"])
        if "window" in state:
            self.window.load_state_dict(state["window"])
        if "trend" in state:
            self.trend.load_state_dict(state["trend"])
        if "slow_trend" in state:
            self.slow_trend.load_state_dict(state["slow_trend"])
        self.best_lr = state.get("best_lr")
        self._rollback_loss = state.get("rollback_loss")
        # The best point's tensors are not serialized. Forgetting its trend value
        # makes the next poll capture a fresh best, so a blow-up soon after a
        # resume has somewhere to go back to.
        self.best_trend = math.inf
        self._best = None
        self._best_trends = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(optimizer={type(self.optimizer).__name__}, "
            f"lr={self.lr}, multiplier={self.multiplier}, "
            f"lr_min={self.lr_min}, lr_max={self.lr_max})"
        )


class EfficientRelativePollingSGD(EfficientRelativePollingOptimizer):
    """:class:`EfficientRelativePollingOptimizer` over plain SGD.

    Args:
        params: parameters to optimize, or the model itself (in which case its
            buffers are tracked automatically).
        lr: the initial learning rate -- the one rate the user chooses.
        module: see :class:`EfficientRelativePollingOptimizer`.

    Remaining keyword arguments configure either :class:`torch.optim.SGD`
    (``momentum``, ``weight_decay``, ``nesterov``, ``dampening``) or the polling
    (``multiplier``, ``lr_min``, ``lr_max``, ``spike_z``, ``blowup_z``,
    ``trend_betas``, ``best_beta``, ``warmup``, ``max_reach``, ``rollback_loss``,
    ``rollback_factor``).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        *,
        module: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        import torch

        params, module = _split_module(params, module)
        sgd_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _SGD_KEYS}
        super().__init__(
            torch.optim.SGD(params, lr=lr, **sgd_kwargs),
            module=module,
            **kwargs,
        )


class EfficientRelativeEpochPolling:
    """The same method at epoch granularity, driven by :func:`fit`.

    A poll here is a whole epoch: from one snapshot, the epoch is trained once
    per candidate rate, and the trial with the lowest mean training loss over
    its epoch -- the progress the epoch made on the training stream, the
    epoch-level analogue of the batch criterion -- keeps its end state. Between
    poll epochs the current rate is used blind, on the same :class:`Backoff`
    schedule as the per-batch method, with ``k`` counted in epochs. The best
    point is the epoch with the highest validation score; a blind epoch that
    blows up -- a non-finite mean loss, or one above the rollback threshold --
    goes back to it and restarts the schedule, and the poll epoch after a
    restart breaks ties downward.

    Selecting on end-of-epoch validation instead is greedy in a way that
    ratchets the rate down: the epoch starts from a point that was itself the
    best seen, so a rate that barely moves the weights keeps that score while
    any real step risks it. On CIFAR-10 that variant walked from ``1e-2`` to
    ``1e-8`` and stalled at 45% test accuracy; the training-loss criterion is
    what this class uses.

    Pass an instance to ``fit(..., epoch_polling=...)`` together with a plain
    optimizer, whose initial ``lr`` is the one rate the user chooses.

    Args:
        multiplier: spacing ``m`` between neighbouring candidates.
        lr_min: optional floor for the candidates.
        lr_max: optional ceiling for the candidates.
        max_reach: how many multipliers out the window may widen while poll
            epochs come back blind; see :class:`Window`.
        rollback_loss: absolute blow-up threshold on an epoch's mean training
            loss. ``None`` derives it as ``rollback_factor`` times the first
            finite epoch loss.
        rollback_factor: multiplier for the derived threshold.
    """

    def __init__(
        self,
        *,
        multiplier: float = 10.0,
        lr_min: float | None = None,
        lr_max: float | None = None,
        max_reach: int = 6,
        rollback_loss: float | None = None,
        rollback_factor: float = 2.0,
    ) -> None:
        if rollback_loss is not None and not math.isfinite(rollback_loss):
            raise ValueError("rollback_loss must be finite, or None to derive it")
        self.window = Window(multiplier, lr_min, lr_max, max_reach)
        self.rollback_factor = float(rollback_factor)
        self._rollback_loss = None if rollback_loss is None else float(rollback_loss)

        self.backoff = Backoff()
        self.best_score = -math.inf
        self.best_lr: float | None = None

        self._optimizer: Optimizer | None = None
        self._start: StateSnapshot | None = None
        self._end: StateSnapshot | None = None
        self._best: StateSnapshot | None = None

    @property
    def rollback_loss(self) -> float | None:
        """Active blow-up threshold, ``None`` until derived from the first epoch."""
        return self._rollback_loss

    @property
    def multiplier(self) -> float:
        return self.window.multiplier

    @property
    def lr_min(self) -> float | None:
        return self.window.lr_min

    @property
    def lr_max(self) -> float | None:
        return self.window.lr_max

    def run_epoch(
        self,
        optimizer: Optimizer,
        module: nn.Module | None,
        train: Callable[[], EpochStats],
        validate: Callable[[], tuple[float, float]],
    ) -> tuple[EpochStats, float, float]:
        """Run one epoch -- a poll epoch or a blind one -- and return its outcome.

        Args:
            optimizer: the plain optimizer whose ``lr`` the controller sets.
            module: the model, so buffers travel with the snapshots.
            train: trains one epoch at the rate currently on ``optimizer`` and
                returns its :class:`~efficient_polling_lr_scheduler.training.EpochStats`,
                whose ``loss`` is what a poll epoch selects on.
            validate: scores the current parameters on held-out data,
                returning ``(loss, score)``; the score judges the best point.

        Returns:
            ``(stats, val_loss, val_score)`` for the epoch that was kept. On a
            blow-up, ``stats.rollbacks`` is 1 and the validation numbers are
            those of the restored best point.
        """
        self._bind(optimizer, module)
        centre = self._lr()

        if self.backoff.due():
            kept = self._poll_epoch(centre, train, validate)
            if kept is None:  # every candidate blew up
                return self._restart(replace(self._last_stats, rollbacks=1), validate)
            stats, val_loss, val_score = kept
        else:
            stats = train()
            if self._blew_up(stats.loss):
                return self._restart(replace(stats, rollbacks=1), validate)
            self._resolve_rollback_loss(stats.loss)
            val_loss, val_score = validate()
            self.backoff.blind_step()

        if val_score > self.best_score:
            self._capture_best(val_score)
        return stats, val_loss, val_score

    # -- internals ---------------------------------------------------------

    def _poll_epoch(
        self,
        centre: float,
        train: Callable[[], EpochStats],
        validate: Callable[[], tuple[float, float]],
    ) -> tuple[EpochStats, float, float] | None:
        assert self._start is not None and self._end is not None
        candidates = self.window.candidates(centre, ascending=self.backoff.after_restart)
        self._start.capture()

        winner: tuple[float, EpochStats] | None = None
        scores: list[float] = []
        steps = 0
        for lr in candidates:
            self._start.restore()
            self._apply_lr(lr)
            stats = train()
            self._last_stats = stats
            steps += stats.optimizer_steps
            if self._blew_up(stats.loss):
                scores.append(-math.inf)  # a diverging trial simply loses
                continue
            self._resolve_rollback_loss(stats.loss)
            score = -stats.loss  # the epoch's progress on the training stream
            scores.append(score)
            if winner is None or score > -winner[1].loss:  # strict: earlier wins ties
                winner = (lr, stats)
                self._end.capture()

        if winner is None:
            return None
        lr, stats = winner
        self._end.restore()
        self._apply_lr(lr)
        val_loss, val_score = validate()  # the kept epoch, judged once
        had_signal = max(scores) > min(scores)
        self.backoff.polled(changed=lr != centre, had_signal=had_signal)
        self.window.polled(had_signal)
        stats = replace(stats, polls=stats.batches, optimizer_steps=steps)
        return stats, val_loss, val_score

    def _restart(
        self, stats: EpochStats, validate: Callable[[], tuple[float, float]]
    ) -> tuple[EpochStats, float, float]:
        """Back to the best epoch; the schedule starts over from there."""
        if self._best is not None and self.best_lr is not None:
            self._best.restore()
            self._apply_lr(self.best_lr)
        self.backoff.restart()
        self.window.reset()
        val_loss, val_score = validate()  # at the restored point, so the record stays finite
        return stats, val_loss, val_score

    def _capture_best(self, score: float) -> None:
        assert self._best is not None
        self._best.capture()
        self.best_score = score
        self.best_lr = self._lr()

    def _bind(self, optimizer: Optimizer, module: nn.Module | None) -> None:
        if optimizer is self._optimizer:
            return
        _check_bounds(float(optimizer.param_groups[0]["lr"]), self.lr_min, self.lr_max)
        self._optimizer = optimizer
        self._start = StateSnapshot(optimizer, module)
        self._end = StateSnapshot(optimizer, module)
        self._best = StateSnapshot(optimizer, module)
        self._last_stats = EpochStats(loss=math.nan, score=math.nan, lr=math.nan)

    def _lr(self) -> float:
        assert self._optimizer is not None
        return float(self._optimizer.param_groups[0]["lr"])

    def _apply_lr(self, lr: float) -> None:
        assert self._optimizer is not None
        for group in self._optimizer.param_groups:
            group["lr"] = lr

    def _blew_up(self, loss: float) -> bool:
        if not math.isfinite(loss):
            return True
        return self._rollback_loss is not None and loss > self._rollback_loss

    def _resolve_rollback_loss(self, loss: float) -> None:
        if self._rollback_loss is None and math.isfinite(loss):
            self._rollback_loss = self.rollback_factor * loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "backoff": self.backoff.state_dict(),
            "window": self.window.state_dict(),
            "best_score": self.best_score,
            "best_lr": self.best_lr,
            "rollback_loss": self._rollback_loss,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.backoff.load_state_dict(state["backoff"])
        if "window" in state:
            self.window.load_state_dict(state["window"])
        self.best_lr = state.get("best_lr")
        self._rollback_loss = state.get("rollback_loss")
        # The best epoch's tensors are not serialized; forgetting its score makes
        # the next epoch the new best, so a blow-up has somewhere to go back to.
        self.best_score = -math.inf

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(multiplier={self.multiplier}, "
            f"lr_min={self.lr_min}, lr_max={self.lr_max})"
        )
