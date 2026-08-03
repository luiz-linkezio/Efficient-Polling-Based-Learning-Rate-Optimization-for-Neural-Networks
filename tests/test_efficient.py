from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from conftest import TinyNet
from efficient_polling_lr_scheduler import (
    EfficientPollingOptimizer,
    EfficientPollingSGD,
    PollingSGD,
    make_closure,
)


class Scripted:
    """Drives an optimizer with scores and losses the test dictates.

    ``winner`` is the learning rate that should win the next poll; ``loss`` is
    what the closure reports at the current parameters. Both are writable
    mid-run, which is how the tests provoke a selection change or a loss spike.
    """

    def __init__(self, model, optimizer, inputs, winner: float, loss: float = 1.0) -> None:
        self.model = model
        self.optimizer = optimizer
        self.inputs = inputs
        self.winner = winner
        self.loss = loss
        self.signal = True

    def __call__(self):
        lr = float(self.optimizer.param_groups[0]["lr"])
        # Real gradients, but a reported loss value the test controls: subtracting
        # the detached value leaves the graph intact and pins the number.
        raw = self.model(self.inputs).square().sum()
        loss = raw - raw.detach() + self.loss
        if not self.signal:
            return loss, 0.0
        return loss, 1.0 if lr == self.winner else 0.0


@pytest.fixture
def scripted(batch):
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(model, lr=1e-3, candidate_lrs=(1e-4, 1e-3, 1e-2))
    return poller, Scripted(model, poller.optimizer, inputs, winner=1e-3), model


def test_first_batch_polls(scripted) -> None:
    poller, closure, _ = scripted
    assert poller.step(closure).polled


def test_interval_doubles_while_the_choice_is_stable(scripted) -> None:
    poller, closure, _ = scripted
    intervals = []
    for _ in range(40):
        info = poller.step(closure)
        if info.polled:
            intervals.append(info.poll_interval)

    assert intervals[:5] == [1, 2, 4, 8, 16]


def test_interval_is_capped(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(model, lr=1e-3, candidate_lrs=(1e-4, 1e-3), max_poll_interval=4)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)

    for _ in range(60):
        poller.step(closure)
    assert poller.poll_interval == 4


def test_a_changed_choice_resets_the_interval(scripted) -> None:
    poller, closure, _ = scripted
    for _ in range(20):
        poller.step(closure)
    assert poller.poll_interval > 1

    closure.winner = 1e-2  # the selection moves
    for _ in range(poller.poll_interval + 1):
        info = poller.step(closure)
        if info.polled:
            break

    assert info.polled
    assert info.lr == 1e-2
    assert info.poll_interval == 1


def test_backoff_waits_for_signal(batch) -> None:
    """All-tie polls must not be read as stability, or training stalls at min lr."""
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(model, lr=1e-3, candidate_lrs=(1e-4, 1e-3, 1e-2))
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)
    closure.signal = False  # every candidate ties, as at initialization

    for _ in range(10):
        info = poller.step(closure)
        assert poller.poll_interval == 1
        if info.polled:
            assert not info.had_signal
    assert not poller.signal_seen

    closure.signal = True
    for _ in range(3):
        if poller.step(closure).polled:
            break
    assert poller.signal_seen


def test_blind_steps_reuse_the_last_polled_lr(scripted) -> None:
    poller, closure, _ = scripted
    poller.step(closure)  # polls, selects 1e-3
    info = poller.step(closure)

    assert not info.polled
    assert info.lr == 1e-3
    assert info.optimizer_steps == 1


def test_steady_state_polls_once_every_k_plus_one_batches(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    k_max = 8
    poller = EfficientPollingSGD(
        model, lr=1e-3, candidate_lrs=(1e-4, 1e-3), max_poll_interval=k_max
    )
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)

    for _ in range(100):  # settle into the cap
        poller.step(closure)

    polled = [poller.step(closure).polled for _ in range(k_max + 1)]
    assert sum(polled) == 1


def test_max_poll_interval_zero_matches_the_base_method(batch, loss_fn) -> None:
    inputs, targets = batch
    torch.manual_seed(7)
    base_model = TinyNet()
    torch.manual_seed(7)
    eff_model = TinyNet()

    base = PollingSGD(base_model, lr=1e-3, candidate_lrs=(1e-4, 1e-3, 1e-2))
    eff = EfficientPollingOptimizer(
        torch.optim.SGD(eff_model.parameters(), lr=1e-3),
        candidate_lrs=(1e-4, 1e-3, 1e-2),
        module=eff_model,
        max_poll_interval=0,  # poll every batch
        spike_factor=0.0,  # the base method has no guard
        rollback_loss=1e9,
    )

    for _ in range(15):
        base_info = base.step(make_closure(base_model, loss_fn, inputs, targets))
        eff_info = eff.step(make_closure(eff_model, loss_fn, inputs, targets))
        assert eff_info.polled
        assert eff_info.lr == base_info.lr

    for a, b in zip(base_model.parameters(), eff_model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-7)


@pytest.fixture
def spiky(batch):
    """Tier 2 is only reachable when gamma * EMA sits below the rollback bar.

    That is the regime the method spends most of training in: the loss EMA falls
    well under the fixed rollback threshold, so moderate spikes get polled rather
    than rolled back. Pinning the threshold high isolates the spike tier.
    """
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(
        model, lr=1e-3, candidate_lrs=(1e-4, 1e-3, 1e-2), rollback_loss=1e9
    )
    return poller, Scripted(model, poller.optimizer, inputs, winner=1e-3)


def test_a_loss_spike_forces_a_poll(spiky) -> None:
    poller, closure = spiky
    for _ in range(20):  # back off well past every-batch polling
        poller.step(closure)
    assert poller.poll_interval > 1

    closure.loss = 10.0 * poller.ema_loss  # above gamma * EMA, below rollback
    info = poller.step(closure)

    assert info.spike
    assert info.polled
    assert not info.rolled_back


def test_spike_factor_zero_disables_the_spike_tier(spiky) -> None:
    poller, closure = spiky
    poller.spike_factor = 0.0
    for _ in range(20):
        poller.step(closure)
    assert poller.poll_interval > 1

    closure.loss = 1e5
    info = poller.step(closure)
    assert not info.spike


def test_a_big_enough_spike_is_rolled_back_instead(scripted) -> None:
    """With the paper's fixed threshold, a violent spike hits tier 1 first."""
    poller, closure, _ = scripted
    for _ in range(20):
        poller.step(closure)

    closure.loss = 100.0 * poller.ema_loss
    info = poller.step(closure)

    assert info.rolled_back
    assert not info.polled


def test_rollback_restores_the_last_poll(scripted) -> None:
    poller, closure, model = scripted
    poller.step(closure)  # establishes the checkpoint
    checkpointed = [p.detach().clone() for p in model.parameters()]

    poller.step(closure)  # a blind step moves the weights
    assert not torch.equal(next(model.parameters()), checkpointed[0])

    closure.loss = math.inf
    info = poller.step(closure)

    assert info.rolled_back
    assert info.optimizer_steps == 0
    for param, saved in zip(model.parameters(), checkpointed, strict=True):
        assert torch.equal(param, saved)


def test_rollback_resumes_polling(scripted) -> None:
    poller, closure, _ = scripted
    for _ in range(20):
        poller.step(closure)

    closure.loss = float("nan")
    poller.step(closure)
    assert poller.poll_interval == 1

    closure.loss = 1.0
    assert poller.step(closure).polled


def test_rollback_threshold_is_derived_from_the_first_loss(scripted) -> None:
    poller, closure, _ = scripted
    assert poller.rollback_loss is None

    closure.loss = math.log(10)  # random-guess cross-entropy for ten classes
    poller.step(closure)

    assert poller.rollback_loss == pytest.approx(2.0 * math.log(10))


def test_explicit_rollback_threshold_is_respected(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(model, lr=1e-3, rollback_loss=5.0)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3, loss=1.0)

    assert not poller.step(closure).rolled_back
    closure.loss = 5.5
    assert poller.step(closure).rolled_back


def test_a_non_finite_loss_rolls_back_before_any_threshold_exists(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(model, lr=1e-3)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3, loss=math.inf)

    info = poller.step(closure)
    assert info.rolled_back
    assert poller.rollback_loss is None  # still undecided, no finite loss seen


def test_polls_a_small_fraction_of_batches(loss_fn) -> None:
    """The headline claim, in miniature: most batches must go unpolled."""
    torch.manual_seed(0)
    inputs = torch.randn(256, 16)
    targets = torch.randint(0, 10, (256,))
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))
    poller = EfficientPollingSGD(model, lr=1e-3)
    closure = make_closure(model, loss_fn, inputs, targets)

    polls = sum(poller.step(closure).polled for _ in range(400))

    assert poller.signal_seen
    assert poller.poll_interval == poller.max_poll_interval  # fully backed off
    assert polls / 400 < 0.15


def test_a_criterion_too_coarse_to_discriminate_keeps_polling(batch, loss_fn) -> None:
    """The anti-stall guard, on real data rather than a scripted closure.

    On an 8-sample batch every candidate reaches the same batch accuracy, so no
    poll ever carries signal. Backing off on those ties would freeze the run at
    the smallest candidate, so the schedule deliberately stays at every-other-batch.
    """
    inputs, targets = batch
    model = TinyNet()
    poller = EfficientPollingSGD(model, lr=1e-3)
    closure = make_closure(model, loss_fn, inputs, targets)

    for _ in range(50):
        info = poller.step(closure)
        assert not info.had_signal

    assert not poller.signal_seen
    assert poller.poll_interval == 1


def test_adding_a_param_group_drops_the_stale_checkpoint(scripted) -> None:
    """A new parameter group invalidates the rollback checkpoint, not just the trial one."""
    poller, closure, _ = scripted
    poller.step(closure)  # takes a checkpoint over the original parameters
    assert poller._checkpoint is not None

    extra = nn.Linear(4, 2)
    poller.add_param_group({"params": list(extra.parameters()), "lr": 1e-3})
    assert poller._checkpoint is None

    # A divergence now must not try to restore the old parameter list.
    closure.loss = math.inf
    assert poller.step(closure).rolled_back
    closure.loss = 1.0
    assert poller.step(closure).polled


def test_rejects_a_bad_ema_beta(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="loss_ema_beta"):
        EfficientPollingSGD(model, lr=1e-3, loss_ema_beta=1.0)


def test_rejects_a_negative_max_interval(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="max_poll_interval"):
        EfficientPollingSGD(model, lr=1e-3, max_poll_interval=-1)


def test_sgd_and_schedule_kwargs_are_routed(model: TinyNet) -> None:
    poller = EfficientPollingSGD(
        model, lr=1e-3, momentum=0.9, weight_decay=1e-4, max_poll_interval=16
    )
    assert poller.optimizer.param_groups[0]["momentum"] == 0.9
    assert poller.optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert poller.max_poll_interval == 16


def test_state_dict_preserves_the_schedule(scripted) -> None:
    poller, closure, _ = scripted
    for _ in range(20):
        poller.step(closure)
    saved = poller.state_dict()

    restored = EfficientPollingSGD(TinyNet(), lr=1e-3, candidate_lrs=(1e-4, 1e-3, 1e-2))
    restored.load_state_dict(saved)

    assert restored.poll_interval == poller.poll_interval
    assert restored.since_poll == poller.since_poll
    assert restored.last_lr == poller.last_lr
    assert restored.signal_seen == poller.signal_seen
    assert restored.ema_loss == pytest.approx(poller.ema_loss)
    assert restored.rollback_loss == pytest.approx(poller.rollback_loss)


# -- trigger ablation ------------------------------------------------------


def test_fixed_trigger_ignores_a_stable_selection(batch) -> None:
    """The backoff variant would double its interval here; this one must not."""
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(
        model, lr=1e-3, candidate_lrs=(1e-4, 1e-3), max_poll_interval=4, trigger="fixed"
    )
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)

    polled = [poller.step(closure).polled for _ in range(21)]

    assert poller.poll_interval == 4
    assert polled == [i % 5 == 0 for i in range(21)]


def test_fixed_trigger_returns_to_its_interval_after_a_rollback(batch) -> None:
    """A rollback must not strand the fixed variant at one poll per batch."""
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(
        model,
        lr=1e-3,
        candidate_lrs=(1e-4, 1e-3),
        max_poll_interval=4,
        trigger="fixed",
        rollback_loss=10.0,
    )
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)
    poller.step(closure)

    closure.loss = 1e3
    assert poller.step(closure).rolled_back
    closure.loss = 1.0

    polled = [poller.step(closure).polled for _ in range(11)]
    assert polled[0], "the batch after a rollback is polled by every trigger"
    assert polled == [i % 5 == 0 for i in range(11)]


def test_random_trigger_polls_at_about_its_probability(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(
        model,
        lr=1e-3,
        candidate_lrs=(1e-4, 1e-3),
        trigger="random",
        poll_probability=0.2,
        poll_seed=0,
        spike_factor=0.0,  # spikes would add polls the trigger did not ask for
    )
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)

    polled = [poller.step(closure).polled for _ in range(1000)]

    assert polled[0], "the first batch is polled, so there is a selection to reuse"
    assert 0.15 < sum(polled) / len(polled) < 0.25


def test_random_trigger_is_reproducible_and_seed_dependent(batch) -> None:
    inputs, _ = batch

    def run(poll_seed: int) -> list[bool]:
        torch.manual_seed(7)
        model = TinyNet()
        poller = EfficientPollingSGD(
            model,
            lr=1e-3,
            candidate_lrs=(1e-4, 1e-3),
            trigger="random",
            poll_probability=0.3,
            poll_seed=poll_seed,
            spike_factor=0.0,
        )
        closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)
        return [poller.step(closure).polled for _ in range(200)]

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_random_trigger_does_not_disturb_the_global_rng(batch) -> None:
    """Batch order must not depend on which trigger the run uses."""
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientPollingSGD(
        model, lr=1e-3, candidate_lrs=(1e-4, 1e-3), trigger="random", poll_probability=0.5
    )
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)

    torch.manual_seed(11)
    for _ in range(50):
        poller.step(closure)
    after = torch.rand(3)

    torch.manual_seed(11)
    assert torch.equal(after, torch.rand(3))


def test_random_trigger_state_dict_resumes_the_same_draws(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    kwargs = dict(
        lr=1e-3,
        candidate_lrs=(1e-4, 1e-3),
        trigger="random",
        poll_probability=0.3,
        spike_factor=0.0,
    )
    poller = EfficientPollingSGD(model, **kwargs)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)
    for _ in range(30):
        poller.step(closure)

    restored = EfficientPollingSGD(TinyNet(), **kwargs)
    restored.load_state_dict(poller.state_dict())

    assert [restored._rng.random() for _ in range(10)] == [poller._rng.random() for _ in range(10)]


def test_rejects_an_unknown_trigger(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="trigger"):
        EfficientPollingSGD(model, lr=1e-3, trigger="backoff_")


def test_rejects_an_out_of_range_poll_probability(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="poll_probability"):
        EfficientPollingSGD(model, lr=1e-3, trigger="random", poll_probability=0.0)
