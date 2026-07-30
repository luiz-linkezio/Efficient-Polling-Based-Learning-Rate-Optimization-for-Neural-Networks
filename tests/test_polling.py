from __future__ import annotations

import pytest
import torch
from torch import nn

from conftest import BatchNormNet, TinyNet
from efficient_polling import (
    PollingOptimizer,
    PollingSGD,
    accuracy,
    default_candidate_lrs,
    make_closure,
)


def scripted_closure(model, optimizer, inputs, scores: dict[float, float], calls: list[float]):
    """A closure whose score is dictated by the learning rate under trial.

    The poller sets ``lr`` on the parameter groups before each trial step, so
    reading it back is how a test scripts which candidate ought to win.
    """

    def closure():
        loss = model(inputs).square().sum()
        lr = float(optimizer.param_groups[0]["lr"])
        calls.append(lr)
        return loss, scores.get(lr, 0.0)

    return closure


def test_picks_the_highest_scoring_candidate(model: TinyNet, batch) -> None:
    inputs, _ = batch
    candidates = (1e-4, 1e-3, 1e-2)
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=candidates)
    closure = scripted_closure(model, poller.optimizer, inputs, {1e-3: 1.0}, [])

    info = poller.step(closure)

    assert info.lr == 1e-3
    assert info.polled
    assert info.had_signal


def test_ties_favour_the_smallest_learning_rate(model: TinyNet, batch) -> None:
    inputs, _ = batch
    candidates = (1e-4, 1e-3, 1e-2)
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=candidates)
    closure = scripted_closure(model, poller.optimizer, inputs, {}, [])  # every score 0.0

    info = poller.step(closure)

    assert info.lr == 1e-4
    assert not info.had_signal


def test_candidates_are_polled_in_ascending_order(model: TinyNet, batch) -> None:
    inputs, _ = batch
    calls: list[float] = []
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=(1e-2, 1e-4, 1e-3))
    poller.step(scripted_closure(model, poller.optimizer, inputs, {}, calls))

    assert poller.candidate_lrs == (1e-4, 1e-3, 1e-2)
    # first call is the gradient evaluation at the current lr, then the trials
    assert calls[1:4] == [1e-4, 1e-3, 1e-2]


def test_applied_step_matches_the_winning_learning_rate(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    winner = 1e-2
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=(1e-4, 1e-3, winner))

    before = [p.detach().clone() for p in model.parameters()]
    poller.optimizer.zero_grad(set_to_none=True)
    loss_fn(model(inputs), targets).backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]

    closure = scripted_closure(model, poller.optimizer, inputs, {winner: 1.0}, [])
    result = poller.poll(closure)

    assert result.lr == winner
    for param, start, grad in zip(model.parameters(), before, grads, strict=True):
        assert torch.allclose(param, start - winner * grad, atol=1e-7)


def test_gradient_is_computed_once_per_batch(model: TinyNet, batch) -> None:
    inputs, _ = batch
    calls: list[float] = []
    candidates = (1e-4, 1e-3, 1e-2)
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=candidates)
    poller.step(scripted_closure(model, poller.optimizer, inputs, {}, calls))

    # one gradient evaluation plus one scoring pass per candidate
    assert len(calls) == 1 + len(candidates)


def test_reports_the_cost_of_a_poll(model: TinyNet, batch) -> None:
    inputs, _ = batch
    candidates = (1e-4, 1e-3, 1e-2, 1e-1)
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=candidates)
    info = poller.step(scripted_closure(model, poller.optimizer, inputs, {}, []))

    # N trial steps plus re-applying the winner
    assert info.optimizer_steps == len(candidates) + 1


def test_reports_pre_and_post_step_metrics(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    poller = PollingSGD(model, lr=1e-1, candidate_lrs=(1e-3, 1e-2, 1e-1))
    info = poller.step(make_closure(model, loss_fn, inputs, targets))

    assert info.loss > 0.0
    assert info.post_loss is not None and info.post_score is not None
    assert 0.0 <= info.score <= 1.0
    assert 0.0 <= info.post_score <= 1.0


def test_default_candidates_span_the_paper_range() -> None:
    assert default_candidate_lrs(1e-3) == pytest.approx((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))


def test_default_candidates_come_from_the_optimizer_lr(model: TinyNet) -> None:
    poller = PollingSGD(model, lr=1e-2)
    assert poller.candidate_lrs == pytest.approx((1e-4, 1e-3, 1e-2, 1e-1, 1e0))


def test_polling_reduces_the_loss(batch, loss_fn) -> None:
    """The real criterion on a real batch: repeated polling must make progress."""
    inputs, targets = batch
    model = TinyNet()
    poller = PollingSGD(model, lr=1e-3)

    closure = make_closure(model, loss_fn, inputs, targets, accuracy)
    first_loss = poller.step(closure).loss
    for _ in range(29):
        poller.step(closure)

    with torch.no_grad():
        final_loss = float(loss_fn(model(inputs), targets))
    assert final_loss < first_loss


def test_works_with_momentum(batch, loss_fn) -> None:
    inputs, targets = batch
    model = TinyNet()
    poller = PollingSGD(model, lr=1e-2, momentum=0.9)

    for _ in range(5):
        info = poller.step(make_closure(model, loss_fn, inputs, targets))
        assert info.lr in poller.candidate_lrs
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_works_with_adam(batch, loss_fn) -> None:
    inputs, targets = batch
    model = TinyNet()
    poller = PollingOptimizer(torch.optim.Adam(model.parameters(), lr=1e-3), module=model)

    for _ in range(5):
        poller.step(make_closure(model, loss_fn, inputs, targets))
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_trials_do_not_leak_batchnorm_statistics(batch, loss_fn) -> None:
    inputs, targets = batch
    net = BatchNormNet()
    net.train()
    poller = PollingSGD(net, lr=1e-2)

    poller.step(make_closure(net, loss_fn, inputs, targets))

    # one polled batch means exactly one tracked batch, not one per candidate
    assert int(net.bn.num_batches_tracked) == 1


def test_rejects_a_detached_loss(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    poller = PollingSGD(model, lr=1e-3)

    def bad_closure():
        with torch.no_grad():
            return loss_fn(model(inputs), targets), 0.0

    with pytest.raises(RuntimeError, match="requires grad"):
        poller.step(bad_closure)


def test_rejects_a_non_optimizer() -> None:
    with pytest.raises(TypeError, match="torch.optim.Optimizer"):
        PollingOptimizer("not an optimizer")  # type: ignore[arg-type]


def test_rejects_empty_candidates(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PollingSGD(model, lr=1e-3, candidate_lrs=())


def test_rejects_negative_candidates(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PollingSGD(model, lr=1e-3, candidate_lrs=(-1e-3, 1e-3))


def test_exposes_the_optimizer_protocol(model: TinyNet) -> None:
    poller = PollingSGD(model, lr=1e-3)
    assert poller.param_groups is poller.optimizer.param_groups
    assert poller.lr == 1e-3
    poller.zero_grad()

    extra = nn.Linear(2, 2)
    poller.add_param_group({"params": list(extra.parameters()), "lr": 1e-3})
    assert len(poller.param_groups) == 2


def test_state_dict_round_trip(model: TinyNet, batch, loss_fn) -> None:
    inputs, targets = batch
    poller = PollingSGD(model, lr=1e-3, momentum=0.9, candidate_lrs=(1e-4, 1e-3))
    poller.step(make_closure(model, loss_fn, inputs, targets))
    saved = poller.state_dict()

    restored = PollingSGD(TinyNet(), lr=1e-3, momentum=0.9, candidate_lrs=(1.0,))
    restored.load_state_dict(saved)

    assert restored.candidate_lrs == (1e-4, 1e-3)
