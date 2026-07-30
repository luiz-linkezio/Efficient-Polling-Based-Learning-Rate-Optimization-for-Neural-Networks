"""Exact save/restore of everything a trial optimizer step can mutate."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["StateSnapshot"]


def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in state.items()
    }


class StateSnapshot:
    """A pristine copy of the state a candidate step departs from.

    Polling only means something if every candidate learning rate starts from
    an *identical* pre-step condition, so a trial step has to be undone
    exactly. Three things move when an optimizer steps:

    * the parameters themselves;
    * the module's buffers (BatchNorm running statistics and friends), which
      the trial forward pass mutates in training mode -- only tracked when a
      ``module`` is supplied;
    * the optimizer's own state (momentum buffers, Adam moments, step
      counters), including state that did not exist before the trial.

    Gradients are deliberately *not* saved: every candidate reuses the single
    gradient computed once at the polled point, exactly as the method
    prescribes.
    """

    def __init__(self, optimizer: Optimizer, module: nn.Module | None = None) -> None:
        self.optimizer = optimizer
        self.module = module
        self._params: list[torch.Tensor] = []
        self._param_values: list[torch.Tensor] = []
        self._buffers: list[torch.Tensor] = []
        self._buffer_values: list[torch.Tensor] = []
        self._state: dict[torch.Tensor, dict[str, Any]] = {}
        self._lrs: list[float] = []
        self.capture()

    def capture(self) -> None:
        """Overwrite the snapshot with the optimizer's current state."""
        optimizer = self.optimizer
        self._params = [p for group in optimizer.param_groups for p in group["params"]]
        self._buffers = list(self.module.buffers()) if self.module is not None else []

        with torch.no_grad():
            self._param_values = [p.detach().clone() for p in self._params]
            self._buffer_values = [b.detach().clone() for b in self._buffers]

        # ``optimizer.state`` is a defaultdict; membership tests must not create
        # entries, hence the explicit ``in`` check rather than a plain lookup.
        self._state = {
            p: _clone_state(optimizer.state[p]) for p in self._params if p in optimizer.state
        }
        self._lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def restore(self) -> None:
        """Put the optimizer back exactly where :meth:`capture` found it."""
        optimizer = self.optimizer

        with torch.no_grad():
            for param, value in zip(self._params, self._param_values, strict=True):
                param.detach().copy_(value)
            for buffer, value in zip(self._buffers, self._buffer_values, strict=True):
                buffer.detach().copy_(value)

        for param in self._params:
            saved = self._state.get(param)
            if saved is None:
                # State the trial step created out of nothing (a first-step
                # momentum buffer, say) has to disappear again.
                optimizer.state.pop(param, None)
            else:
                # Hand out a copy: the live state must never alias the snapshot,
                # or the next step would silently corrupt it.
                optimizer.state[param] = _clone_state(saved)

        # strict=True: a parameter group added since capture makes this snapshot
        # stale, and silently restoring a prefix would corrupt training.
        for group, lr in zip(optimizer.param_groups, self._lrs, strict=True):
            group["lr"] = lr
