from __future__ import annotations

import pytest
import torch
from torch import nn

from conftest import BatchNormNet, TinyNet
from efficient_polling_lr_scheduler import SPSSGD, ArmijoSGD, make_closure


def test_sps_sets_the_polyak_step(model: TinyNet, batch, loss_fn: nn.Module) -> None:
    inputs, targets = batch
    optimizer = SPSSGD(model, lr=1e-3, max_lr=1e6)

    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))

    grad_sq = sum(float(p.grad.square().sum()) for p in model.parameters())
    assert info.lr == pytest.approx(info.loss / grad_sq, rel=1e-5)
    assert info.optimizer_steps == 1
    assert not info.polled


def test_sps_respects_the_cap(model: TinyNet, batch, loss_fn: nn.Module) -> None:
    inputs, targets = batch
    optimizer = SPSSGD(model, lr=1e-3, max_lr=1e-4)

    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))

    assert info.lr == 1e-4


def test_sps_reports_the_pre_step_loss(model: TinyNet, batch, loss_fn: nn.Module) -> None:
    """The reported loss must be the one whose gradient drove the step."""
    inputs, targets = batch
    optimizer = SPSSGD(model, lr=1e-3)

    with torch.no_grad():
        expected = float(loss_fn(model(inputs), targets))
    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))

    assert info.loss == pytest.approx(expected, rel=1e-6)


def test_armijo_accepts_a_step_that_decreases_enough(
    model: TinyNet, batch, loss_fn: nn.Module
) -> None:
    inputs, targets = batch
    optimizer = ArmijoSGD(model, lr_max=1e-3, alpha=1e-4, beta=0.5, max_iters=10)

    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))

    # A tiny step on a smooth loss satisfies the condition immediately.
    assert info.lr == 1e-3
    assert info.optimizer_steps == 1
    assert info.post_loss <= info.loss


def test_armijo_backtracks_when_the_first_step_overshoots(
    model: TinyNet, batch, loss_fn: nn.Module
) -> None:
    inputs, targets = batch
    optimizer = ArmijoSGD(model, lr_max=5e3, alpha=1e-4, beta=0.5, max_iters=20)

    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))

    assert info.lr < 5e3
    assert info.optimizer_steps > 1
    assert info.had_signal


def test_armijo_keeps_the_last_trial_when_the_budget_runs_out(
    model: TinyNet, batch, loss_fn: nn.Module
) -> None:
    inputs, targets = batch
    optimizer = ArmijoSGD(model, lr_max=1e6, alpha=1e-4, beta=0.5, max_iters=3)

    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))

    assert info.optimizer_steps == 3
    assert info.lr == pytest.approx(1e6 * 0.5**2)


def test_armijo_reports_the_pre_step_loss(model: TinyNet, batch, loss_fn: nn.Module) -> None:
    inputs, targets = batch
    optimizer = ArmijoSGD(model, lr_max=1e-2)

    with torch.no_grad():
        expected = float(loss_fn(model(inputs), targets))
    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))

    assert info.loss == pytest.approx(expected, rel=1e-6)
    # post_loss is the trial the search settled on, and is *not* what fit() logs.
    assert info.post_loss != info.loss


def test_armijo_trials_all_depart_from_the_same_point(
    model: TinyNet, batch, loss_fn: nn.Module
) -> None:
    """Every rejected trial must be undone exactly.

    After a search that backtracks several times, the parameters must equal one
    plain SGD step at the accepted learning rate taken from the *original*
    point -- not a compounding of the trials.
    """
    inputs, targets = batch
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = ArmijoSGD(model, lr_max=5e3, beta=0.5, max_iters=20)

    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))
    assert info.optimizer_steps > 1, "this test is only meaningful if it backtracked"

    after = [p.detach().clone() for p in model.parameters()]
    grads = [p.grad.detach().clone() for p in model.parameters()]
    for start, end, grad in zip(before, after, grads, strict=True):
        torch.testing.assert_close(end, start - info.lr * grad)


def test_armijo_restores_module_buffers_between_trials(batch, loss_fn: nn.Module) -> None:
    """A trial forward pass moves BatchNorm statistics; the snapshot rolls them
    back, so the count of trials cannot change where the buffers end up."""
    inputs, targets = batch

    def run(max_iters: int) -> list[torch.Tensor]:
        torch.manual_seed(0)
        model = BatchNormNet()
        optimizer = ArmijoSGD(model, lr_max=5e3, beta=0.5, max_iters=max_iters)
        optimizer.step(make_closure(model, loss_fn, inputs, targets))
        return [b.detach().clone() for b in model.buffers()]

    for few, many in zip(run(2), run(12), strict=True):
        torch.testing.assert_close(few, many)


def test_armijo_rejects_an_invalid_beta(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="beta"):
        ArmijoSGD(model, beta=1.5)


def test_armijo_rejects_an_empty_budget(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="max_iters"):
        ArmijoSGD(model, max_iters=0)
