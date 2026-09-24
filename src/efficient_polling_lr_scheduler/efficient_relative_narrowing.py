"""Efficient Relative Narrowing Polling: the relative window, with a jump that narrows.

Efficient Relative Polling moves the rate by a whole multiplier at a time: with
``m = 10`` it lives on decades, and when the rate the batch wants lies between
two of them it hops from one to the other and never tries what lies between.
This variant lets the jump ``f`` between the centre and its neighbours adapt,
from what the polls keep showing rather than from any single one of them:

* **the rate is bracketed** when the centre wins, or when the winner reverses --
  up after a move down, or down after a move up. The two are one behaviour:
  with ``{10^1, 10^2, 10^3}``, ``10^2`` winning every time says what an
  oscillation between ``10^1`` and ``10^3`` says. A lasting bracket means a
  plateau, and the jump **narrows**, to look between the candidates;
* **the rate keeps going** when the winner moves the same way as the last move,
  only up or only down. A lasting run means the rate is still far, and the jump
  **widens** back, on both sides, to search a wider range.

Each behaviour has a patience, counted in polls: ``p_n`` for narrowing and
``p_w`` for widening, both starting at ``patience``. A poll of one behaviour
takes a whole poll off its own patience and ``break_discount`` of a poll off the
other's, since a poll that breaks a trend is weak evidence against it: it slows
the countdown without giving back what was spent, so a stray poll cannot undo a
plateau that is forming. When a patience runs out the jump moves one step, unless
it is already at ``m`` or at the cap ``max_narrowings`` sets, and both patiences
start again from ``patience``. The patience is a different counter from the poll
interval ``k`` of the backoff, which is untouched: ``k`` decides when to poll,
``p_n`` and ``p_w`` how finely.

Nothing caps the narrowings by default. A plateau that lasts keeps narrowing the
jump, and what stops it is the polls themselves: candidates too close for the
criterion to tell apart tie, and a tie on a narrowed jump counts toward
widening. ``max_narrowings`` sets a cap when one is wanted. The one limit that
always holds is numerical: a narrowing that would put the neighbours within
rounding of the centre does not happen, since the poll would try one rate.

A step is soft: a narrowing takes a fraction ``narrowing`` off the jump,
measured in orders of magnitude, ``f -> f ** (1 - narrowing)``, and a widening
puts one such step back, never beyond ``m``. With the default ``0.5`` one
narrowing from ``m = 10`` makes the next neighbour ``X * 10 ** 0.5``, the
geometric middle between the two decades; with ``0`` the jump never moves and
the method is Efficient Relative Polling, apart from the rounding at the bounds
described below.

A tie -- every candidate scoring the same -- on a narrowed jump says the jump is
too fine for the criterion to tell apart, and counts as a poll for widening. A
tie at ``m`` widens the window at once, as in Efficient Relative Polling, which
is how a run leaves the initialization plateau. A poll whose centre sits on
``lr_min`` or ``lr_max`` has only two candidates, so the centre winning there
says nothing about a bracket and counts for neither behaviour, and neither does
the poll that ends a blind stretch, which only brings the window back to ``m``.
A restart goes back to the full multiplier with fresh patiences, and everything
else -- the backoff, the trend, the restarts -- is Efficient Relative Polling
unchanged.

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
# A patience at or below this has run out. A discount that binary floats cannot
# hold exactly, such as 0.6, leaves ~1e-16 where the count is exactly zero.
_EXHAUSTED = 1e-9

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
    effect, and is only ever above zero while ``reach`` is one. ``depth`` moves
    when a patience runs out: ``narrow_patience`` (``p_n``) or
    ``widen_patience`` (``p_w``); see the module docstring.

    Args:
        multiplier: the widest jump outside a blind stretch, ``m``.
        lr_min: optional floor the candidates fold into.
        lr_max: optional ceiling the candidates fold into.
        max_reach: see :class:`Window`.
        narrowing: fraction of the jump, in orders of magnitude, one narrowing
            takes off. ``0.5`` puts the next neighbour on the geometric middle
            between the two rates of the previous jump; ``0`` never narrows.
        max_narrowings: optional cap on the narrowings that can pile up, so the
            jump never gets finer than
            ``multiplier ** ((1 - narrowing) ** max_narrowings)``. ``None``, the
            default, sets none: the jump narrows as long as the polls keep
            bracketing the rate, short of the neighbours rounding onto the centre.
        patience: polls of one behaviour -- the rate bracketed, or the rate
            running one way -- before the jump narrows or widens a step.
        break_discount: what a poll of the other behaviour takes off a
            patience, as a fraction of a poll, in ``[0, 1)``. ``0`` pauses the
            countdown on a break; it stays below one, since a break is weaker
            evidence than a poll of the behaviour itself.
    """

    def __init__(
        self,
        multiplier: float = 10.0,
        lr_min: float | None = None,
        lr_max: float | None = None,
        max_reach: int = 6,
        narrowing: float = 0.5,
        max_narrowings: int | None = None,
        patience: int = 8,
        break_discount: float = 0.5,
    ) -> None:
        super().__init__(multiplier, lr_min, lr_max, max_reach)
        if not (math.isfinite(narrowing) and 0.0 <= narrowing < 1.0):
            raise ValueError(f"narrowing must be in [0, 1), got {narrowing}")
        if max_narrowings is not None and max_narrowings < 0:
            raise ValueError(f"max_narrowings must be at least 0 or None, got {max_narrowings}")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        if not (math.isfinite(break_discount) and 0.0 <= break_discount < 1.0):
            raise ValueError(f"break_discount must be in [0, 1), got {break_discount}")
        self.narrowing = float(narrowing)
        self.max_narrowings = None if max_narrowings is None else int(max_narrowings)
        self.patience = int(patience)
        self.break_discount = float(break_discount)
        self.depth = 0
        # Direction of the last poll that moved the rate: +1 up, -1 down, 0 none yet.
        self.last_move = 0
        self.narrow_patience = float(self.patience)
        self.widen_patience = float(self.patience)

    @property
    def factor(self) -> float:
        if self.depth == 0:
            return super().factor
        return self.multiplier ** ((1.0 - self.narrowing) ** self.depth)

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
        """Count this poll toward a behaviour, and move the jump if a patience runs out.

        Args:
            had_signal: the candidates did not all score the same.
            centre: the rate the poll was centred on.
            winner: the rate the poll chose. Without ``centre`` and ``winner``
                the window only reacts to the signal, as :class:`Window` does.
        """
        if not had_signal:
            if self.depth > 0:
                self._count(widen=True)  # too fine to tell the candidates apart
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
                self._count(widen=False)
        elif move == -self.last_move:
            self._count(widen=False)
        elif move == self.last_move:
            self._count(widen=True)
        if move:
            self.last_move = move

    def _count(self, widen: bool) -> None:
        """One poll of a behaviour: a whole poll off its patience, a discount off the other's."""
        if widen:
            self.widen_patience -= 1.0
            self.narrow_patience -= self.break_discount
        else:
            self.narrow_patience -= 1.0
            self.widen_patience -= self.break_discount
        # The behaviour this poll showed goes first when both run out together.
        ran_out = {
            True: self.widen_patience <= _EXHAUSTED,
            False: self.narrow_patience <= _EXHAUSTED,
        }
        for side in (widen, not widen):
            if ran_out[side]:
                self._step(widen=side)
                return

    def _step(self, widen: bool) -> None:
        if widen:
            self.depth = max(0, self.depth - 1)
        elif self._can_narrow():
            self.depth += 1
        self._refill()

    def _can_narrow(self) -> bool:
        if self.narrowing == 0.0:
            return False
        if self.max_narrowings is not None and self.depth >= self.max_narrowings:
            return False
        # Past this the neighbours would round onto the centre: a poll of one rate.
        finer = self.multiplier ** ((1.0 - self.narrowing) ** (self.depth + 1))
        return not _same(finer, 1.0)

    def _refill(self) -> None:
        self.narrow_patience = float(self.patience)
        self.widen_patience = float(self.patience)

    def reset(self) -> None:
        super().reset()
        self.depth = 0
        self.last_move = 0
        self._refill()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            depth=self.depth,
            last_move=self.last_move,
            narrow_patience=self.narrow_patience,
            widen_patience=self.widen_patience,
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        # A Window's state has none of these, and loads as a jump at the full
        # multiplier with fresh patiences.
        self.depth = int(state.get("depth", 0))
        self.last_move = int(state.get("last_move", 0))
        self.narrow_patience = float(state.get("narrow_patience", self.patience))
        self.widen_patience = float(state.get("widen_patience", self.patience))


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
        max_narrowings: optional cap on the narrowings that can pile up, none by
            default; see :class:`NarrowingWindow`.
        patience: polls of one behaviour before the jump moves a step; see
            :class:`NarrowingWindow`.
        break_discount: what a poll that breaks a behaviour takes off its
            patience, as a fraction of a poll; see :class:`NarrowingWindow`.

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
        max_narrowings: int | None = None,
        patience: int = 8,
        break_discount: float = 0.5,
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
            patience,
            break_discount,
        )

    @property
    def narrowing(self) -> float:
        return self.window.narrowing

    @property
    def max_narrowings(self) -> int | None:
        return self.window.max_narrowings

    @property
    def patience(self) -> int:
        return self.window.patience

    @property
    def break_discount(self) -> float:
        return self.window.break_discount

    def _update_window(self, centre: float, poll: PollResult) -> None:
        self.window.polled(poll.had_signal, centre=centre, winner=poll.lr)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(optimizer={type(self.optimizer).__name__}, "
            f"lr={self.lr}, multiplier={self.multiplier}, narrowing={self.narrowing}, "
            f"max_narrowings={self.max_narrowings}, patience={self.patience}, "
            f"break_discount={self.break_discount}, lr_min={self.lr_min}, lr_max={self.lr_max})"
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
    (``narrowing``, ``max_narrowings``, ``patience``, ``break_discount`` and those of
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
