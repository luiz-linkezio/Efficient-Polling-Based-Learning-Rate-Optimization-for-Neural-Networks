"""Efficient Relative Narrowing Polling: the relative window, with a jump that narrows.

Efficient Relative Polling moves the rate by a whole multiplier at a time: with
``m = 10`` it lives on decades, and when the rate the batch wants lies between
two of them it hops from one to the other and never tries what lies between.
This variant lets the jump itself adapt. After every poll with signal the jump
``f`` of the next one is set from how the winner moved:

* the winner **reversed** -- the rate went up and now comes back down, or the
  other way round -- so the rate the batch wants lies between the two, and the
  jump narrows;
* the **centre won** -- both neighbours lost to the rate in use -- so the rate
  is bracketed, and the jump narrows;
* the winner **kept going** the way it moved last, so the rate is still far,
  and the jump widens back.

A narrowing is soft: it takes a fraction ``narrowing`` off the jump, measured in
orders of magnitude, ``f -> f ** (1 - narrowing)``, and a widening puts one such
step back, so the edges tighten and loosen a step at a time and never beyond
``m``. With the default ``0.5`` one narrowing from ``m = 10`` makes the next
neighbour ``X * 10 ** 0.5``, the geometric middle between the two decades; with
``0`` the jump never moves and the method is Efficient Relative Polling, apart
from the rounding at the bounds described below.

A blind poll -- every candidate scoring the same -- first undoes a narrowing, a
jump too fine for the criterion to tell apart, and only then widens beyond ``m``
the way :class:`~efficient_polling_lr_scheduler.efficient_relative.Window` does.
A poll whose centre sits on ``lr_min`` or ``lr_max`` has only two candidates,
so the centre winning there says nothing about a bracket, and it leaves the jump
as it is. A poll right after a blind stretch only brings the window back to
``m``; the narrowing starts from the poll after it. A restart goes back to the
full multiplier, and everything else -- the backoff, the trend, the restarts --
is Efficient Relative Polling unchanged.

A rate reached through a product of jumps drifts by an ulp or two --
``1e-7 * 10**6`` is ``0.09999999999999999``, not ``0.1`` -- and fractional jumps
reach a bound through such products all the time. A neighbour within rounding
of a bound is therefore put on the bound, and a centre within rounding of one
counts as on it, so a bound neither gains a duplicate candidate nor loses its
exception to float noise.
"""

from __future__ import annotations

import math
from typing import Any

from torch import nn
from torch.optim import Optimizer

from .efficient_relative import EfficientRelativePollingOptimizer, Window
from .polling import _SGD_KEYS, PollResult, _split_module

# Relative distance under which two rates are the same rate. Far below any jump
# the window makes, far above the few ulps a product of jumps drifts by.
_ROUNDING = 1e-9

__all__ = [
    "NarrowingWindow",
    "EfficientRelativeNarrowingPollingOptimizer",
    "EfficientRelativeNarrowingPollingSGD",
]


class NarrowingWindow(Window):
    """A :class:`Window` whose jump also narrows between the multiplier's steps.

    The next poll tries ``{X/f, X, X*f}`` with
    ``f = multiplier ** (reach * (1 - narrowing) ** depth)``. ``reach`` is the
    blind widening of :class:`Window`; ``depth`` counts the narrowings in
    effect, and is only ever above zero while ``reach`` is one.

    Args:
        multiplier: the widest jump outside a blind stretch, ``m``.
        lr_min: optional floor the candidates fold into.
        lr_max: optional ceiling the candidates fold into.
        max_reach: see :class:`Window`.
        narrowing: fraction of the jump, in orders of magnitude, one narrowing
            takes off. ``0.5`` puts the next neighbour on the geometric middle
            between the two rates of the previous jump; ``0`` never narrows.
        max_narrowings: narrowings that can pile up, so the jump never gets
            finer than ``multiplier ** ((1 - narrowing) ** max_narrowings)``.
    """

    def __init__(
        self,
        multiplier: float = 10.0,
        lr_min: float | None = None,
        lr_max: float | None = None,
        max_reach: int = 6,
        narrowing: float = 0.5,
        max_narrowings: int = 3,
    ) -> None:
        super().__init__(multiplier, lr_min, lr_max, max_reach)
        if not (math.isfinite(narrowing) and 0.0 <= narrowing < 1.0):
            raise ValueError(f"narrowing must be in [0, 1), got {narrowing}")
        if max_narrowings < 0:
            raise ValueError(f"max_narrowings must be at least 0, got {max_narrowings}")
        self.narrowing = float(narrowing)
        self.max_narrowings = int(max_narrowings)
        self.depth = 0
        # Direction of the last poll that moved the rate: +1 up, -1 down, 0 none yet.
        self.last_move = 0

    @property
    def factor(self) -> float:
        if self.depth == 0:
            return super().factor
        return self.multiplier ** ((1.0 - self.narrowing) ** self.depth)

    @property
    def finest(self) -> float:
        """The narrowest jump the window can reach."""
        if self.narrowing == 0.0:
            return self.multiplier
        return self.multiplier ** ((1.0 - self.narrowing) ** self.max_narrowings)

    def at_bound(self, centre: float) -> bool:
        """Whether ``centre`` sits on a bound, which folds one neighbour into it."""
        return any(
            bound is not None and _same(centre, bound) for bound in (self.lr_min, self.lr_max)
        ) or (
            (self.lr_max is not None and centre > self.lr_max)
            or (self.lr_min is not None and centre < self.lr_min)
        )

    def candidates(self, centre: float, ascending: bool) -> tuple[float, ...]:
        """As :meth:`Window.candidates`, with the bounds held against rounding.

        A neighbour that folds past a bound or lands within rounding of it is
        put on the bound, and a neighbour within rounding of the centre is
        dropped, so the centre keeps its own value and a poll never tries the
        same rate twice.
        """
        factor = self.factor
        lower, upper = self._fold(centre / factor), self._fold(centre * factor)
        order = (lower, centre, upper) if ascending else (centre, lower, upper)
        candidates: list[float] = []
        for lr in order:
            if lr != centre and _same(lr, centre):
                continue
            if not any(_same(lr, kept) for kept in candidates):
                candidates.append(lr)
        return tuple(candidates)

    def _fold(self, lr: float) -> float:
        for bound in (self.lr_min, self.lr_max):
            if bound is not None and _same(lr, bound):
                return bound
        if self.lr_min is not None:
            lr = max(lr, self.lr_min)
        if self.lr_max is not None:
            lr = min(lr, self.lr_max)
        return lr

    def polled(
        self, had_signal: bool, centre: float | None = None, winner: float | None = None
    ) -> None:
        """Set the jump of the next poll from this one's outcome.

        Args:
            had_signal: the candidates did not all score the same.
            centre: the rate the poll was centred on.
            winner: the rate the poll chose. Without ``centre`` and ``winner``
                the window only reacts to the signal, as :class:`Window` does.
        """
        if not had_signal:
            if self.depth > 0:
                self.depth -= 1
            else:
                super().polled(had_signal)
            return
        if centre is None or winner is None:
            super().polled(had_signal)
            return

        move = 0 if winner == centre else (1 if winner > centre else -1)
        if self.reach > 1:
            super().polled(had_signal)  # a widened window saw: back to one multiplier
        elif move == 0:
            if not self.at_bound(centre):
                self._narrow()
        elif move == -self.last_move:
            self._narrow()
        elif move == self.last_move:
            self.depth = max(0, self.depth - 1)
        if move:
            self.last_move = move

    def _narrow(self) -> None:
        if self.narrowing > 0.0:
            self.depth = min(self.depth + 1, self.max_narrowings)

    def reset(self) -> None:
        super().reset()
        self.depth = 0
        self.last_move = 0

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(depth=self.depth, last_move=self.last_move)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        # A Window's state has neither, and loads as a jump at the full multiplier.
        self.depth = int(state.get("depth", 0))
        self.last_move = int(state.get("last_move", 0))


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_ROUNDING)


class EfficientRelativeNarrowingPollingOptimizer(EfficientRelativePollingOptimizer):
    """Efficient Relative Polling whose jump narrows toward the rate between two candidates.

    See the module docstring for the rule. The poll, the backoff, the trend and
    the restarts are those of
    :class:`~efficient_polling_lr_scheduler.efficient_relative.EfficientRelativePollingOptimizer`;
    only the jump between the centre and its neighbours changes, through
    :class:`NarrowingWindow`.

    Args:
        optimizer: the optimizer whose ``lr`` is polled; its initial ``lr`` is
            the only learning rate the user chooses.
        module: the model, so buffers are restored between trials and by
            restarts.
        narrowing: fraction of the jump, in orders of magnitude, one narrowing
            takes off; see :class:`NarrowingWindow`. ``0`` never narrows,
            which is Efficient Relative Polling with the bounds held against
            rounding.
        max_narrowings: narrowings that can pile up; see :class:`NarrowingWindow`.

    Remaining keyword arguments are those of
    :class:`~efficient_polling_lr_scheduler.efficient_relative.EfficientRelativePollingOptimizer`:
    ``multiplier`` is the widest jump, ``lr_min`` and ``lr_max`` the optional
    bounds, and so on.
    """

    window: NarrowingWindow

    def __init__(
        self,
        optimizer: Optimizer,
        module: nn.Module | None = None,
        *,
        narrowing: float = 0.5,
        max_narrowings: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(optimizer, module, **kwargs)
        window = self.window
        self.window = NarrowingWindow(
            window.multiplier,
            window.lr_min,
            window.lr_max,
            window.max_reach,
            narrowing,
            max_narrowings,
        )

    @property
    def narrowing(self) -> float:
        return self.window.narrowing

    @property
    def max_narrowings(self) -> int:
        return self.window.max_narrowings

    def _update_window(self, centre: float, poll: PollResult) -> None:
        self.window.polled(poll.had_signal, centre=centre, winner=poll.lr)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(optimizer={type(self.optimizer).__name__}, "
            f"lr={self.lr}, multiplier={self.multiplier}, narrowing={self.narrowing}, "
            f"max_narrowings={self.max_narrowings}, lr_min={self.lr_min}, lr_max={self.lr_max})"
        )


class EfficientRelativeNarrowingPollingSGD(EfficientRelativeNarrowingPollingOptimizer):
    """:class:`EfficientRelativeNarrowingPollingOptimizer` over plain SGD.

    Args:
        params: parameters to optimize, or the model itself (in which case its
            buffers are tracked automatically).
        lr: the initial learning rate -- the one rate the user chooses.
        module: see :class:`EfficientRelativeNarrowingPollingOptimizer`.

    Remaining keyword arguments configure either :class:`torch.optim.SGD`
    (``momentum``, ``weight_decay``, ``nesterov``, ``dampening``) or the polling
    (``narrowing``, ``max_narrowings`` and those of
    :class:`~efficient_polling_lr_scheduler.efficient_relative.EfficientRelativePollingSGD`).
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
