from __future__ import annotations

import torch
from torch import nn

from conftest import BatchNormNet, TinyNet
from efficient_polling import StateSnapshot


def test_restores_parameters_exactly(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    snapshot = StateSnapshot(optimizer, model)
    before = [p.detach().clone() for p in model.parameters()]

    loss_fn(model(inputs), targets).backward()
    optimizer.step()
    assert not torch.allclose(next(model.parameters()), before[0])

    snapshot.restore()
    for param, original in zip(model.parameters(), before, strict=True):
        assert torch.equal(param, original)


def test_removes_state_created_by_the_trial_step(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    snapshot = StateSnapshot(optimizer, model)  # no momentum buffers exist yet

    loss_fn(model(inputs), targets).backward()
    optimizer.step()
    assert any("momentum_buffer" in s for s in optimizer.state.values())

    snapshot.restore()
    assert all(not s for s in optimizer.state.values())


def test_restores_existing_momentum_buffers(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn(model(inputs), targets).backward()
    optimizer.step()

    snapshot = StateSnapshot(optimizer, model)
    saved = {p: s["momentum_buffer"].clone() for p, s in optimizer.state.items()}

    optimizer.step()
    snapshot.restore()

    for param, buffer in saved.items():
        assert torch.equal(optimizer.state[param]["momentum_buffer"], buffer)


def test_snapshot_is_not_aliased_by_later_steps(model: TinyNet, batch, loss_fn) -> None:
    """Restoring twice must give the same state both times."""
    inputs, targets = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn(model(inputs), targets).backward()
    optimizer.step()

    snapshot = StateSnapshot(optimizer, model)
    first = [p.detach().clone() for p in model.parameters()]

    optimizer.step()
    optimizer.step()
    snapshot.restore()
    after_first_restore = [p.detach().clone() for p in model.parameters()]

    optimizer.step()
    snapshot.restore()

    for param, a, b in zip(model.parameters(), first, after_first_restore, strict=True):
        assert torch.equal(param, a)
        assert torch.equal(param, b)


def test_restores_module_buffers(batch, loss_fn) -> None:
    inputs, targets = batch
    net = BatchNormNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    net.train()

    snapshot = StateSnapshot(optimizer, net)
    running_mean = net.bn.running_mean.clone()
    batches_tracked = net.bn.num_batches_tracked.clone()

    loss_fn(net(inputs), targets).backward()  # mutates BN running stats
    assert not torch.equal(net.bn.running_mean, running_mean)

    snapshot.restore()
    assert torch.equal(net.bn.running_mean, running_mean)
    assert torch.equal(net.bn.num_batches_tracked, batches_tracked)


def test_restores_learning_rate(model: TinyNet) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    snapshot = StateSnapshot(optimizer, model)
    for group in optimizer.param_groups:
        group["lr"] = 999.0
    snapshot.restore()
    assert optimizer.param_groups[0]["lr"] == 0.25


def test_keeps_gradients(model: TinyNet, batch, loss_fn) -> None:
    """Candidates reuse one gradient, so restore must leave .grad alone."""
    inputs, targets = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn(model(inputs), targets).backward()
    grads = [p.grad.clone() for p in model.parameters()]

    StateSnapshot(optimizer, model).restore()

    for param, grad in zip(model.parameters(), grads, strict=True):
        assert torch.equal(param.grad, grad)


def test_capture_overwrites_previous_snapshot(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    snapshot = StateSnapshot(optimizer, model)

    loss_fn(model(inputs), targets).backward()
    optimizer.step()
    snapshot.capture()
    stepped = [p.detach().clone() for p in model.parameters()]

    optimizer.step()
    snapshot.restore()

    for param, expected in zip(model.parameters(), stepped, strict=True):
        assert torch.equal(param, expected)


def test_works_without_a_module(model: nn.Module, batch, loss_fn) -> None:
    inputs, targets = batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    snapshot = StateSnapshot(optimizer)
    before = [p.detach().clone() for p in model.parameters()]

    loss_fn(model(inputs), targets).backward()
    optimizer.step()
    snapshot.restore()

    for param, original in zip(model.parameters(), before, strict=True):
        assert torch.equal(param, original)
