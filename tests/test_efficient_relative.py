from __future__ import annotations

import math

import pytest

from efficient_polling_lr_scheduler.efficient_relative import Backoff, Trend

# -- Backoff: the TCP-style poll interval ----------------------------------


def stable(backoff: Backoff, times: int = 1) -> None:
    """``times`` polls with signal that kept the rate."""
    for _ in range(times):
        backoff.polled(changed=False, had_signal=True)


def tie(backoff: Backoff, times: int = 1) -> None:
    """``times`` polls where every candidate scored the same."""
    for _ in range(times):
        backoff.polled(changed=False, had_signal=False)


def test_first_step_is_due() -> None:
    assert Backoff().due()


def test_interval_doubles_while_the_choice_is_stable_with_no_cap() -> None:
    backoff = Backoff()
    intervals = []
    for _ in range(12):
        stable(backoff)
        intervals.append(backoff.interval)

    assert intervals == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def test_backoff_waits_for_signal() -> None:
    """All-tie polls carry no information, so they must not stretch the interval."""
    backoff = Backoff()
    tie(backoff, 5)

    assert backoff.interval == 0
    assert not backoff.signal_seen


def test_a_changed_choice_resets_the_interval_to_zero() -> None:
    backoff = Backoff()
    stable(backoff, 5)
    assert backoff.interval == 16

    backoff.polled(changed=True, had_signal=True)
    assert backoff.interval == 0


def test_due_counts_blind_steps_between_polls() -> None:
    backoff = Backoff()
    stable(backoff, 2)  # interval 2
    assert backoff.interval == 2

    assert not backoff.due()
    backoff.blind_step()
    assert not backoff.due()
    backoff.blind_step()
    assert backoff.due()


def test_restart_zeroes_the_interval_and_halves_the_ceiling() -> None:
    backoff = Backoff()
    stable(backoff, 5)
    assert backoff.interval == 16

    backoff.restart()

    assert backoff.interval == 0
    assert backoff.ceiling == 8
    assert backoff.due()
    assert backoff.after_restart


def test_growth_is_linear_above_the_ceiling() -> None:
    """Slow start up to the ceiling, then one blind step more per stable poll."""
    backoff = Backoff()
    stable(backoff, 5)
    backoff.restart()  # ceiling 8
    stable(backoff)  # the restart poll only re-measures

    intervals = []
    for _ in range(7):
        stable(backoff)
        intervals.append(backoff.interval)

    assert intervals == [1, 2, 4, 8, 9, 10, 11]


def test_a_later_restart_halves_the_interval_that_blew_up() -> None:
    """The ceiling remembers the last failure, not the first one."""
    backoff = Backoff()
    stable(backoff, 5)
    backoff.restart()  # ceiling 8
    stable(backoff, 9)  # restart poll, then 1, 2, 4, 8, 9, 10, 11, 12
    assert backoff.interval == 12

    backoff.restart()
    assert backoff.ceiling == 6


def test_a_restart_at_a_tiny_interval_keeps_a_ceiling_of_one() -> None:
    backoff = Backoff()
    backoff.restart()
    assert backoff.ceiling == 1
    stable(backoff)
    intervals = []
    for _ in range(4):
        stable(backoff)
        intervals.append(backoff.interval)
    assert intervals == [1, 2, 3, 4]


def test_the_poll_after_a_restart_clears_the_flag() -> None:
    backoff = Backoff()
    assert not backoff.after_restart  # the very first poll is an ordinary one
    backoff.restart()
    tie(backoff)
    assert not backoff.after_restart


def test_backoff_state_round_trip() -> None:
    backoff = Backoff()
    stable(backoff, 5)
    backoff.restart()
    stable(backoff, 3)
    backoff.blind_step()

    restored = Backoff()
    restored.load_state_dict(backoff.state_dict())

    assert restored.state_dict() == backoff.state_dict()
    assert restored.interval == backoff.interval
    assert restored.ceiling == backoff.ceiling
    assert restored.due() == backoff.due()


# -- Trend: Adam's moments over the loss --------------------------------------


def test_the_first_sample_is_the_mean() -> None:
    trend = Trend()
    trend.update(2.0)
    assert trend.mean == pytest.approx(2.0)
    assert trend.std == pytest.approx(0.0)


def test_a_constant_series_has_zero_spread() -> None:
    trend = Trend()
    for _ in range(200):
        trend.update(3.0)
    assert trend.mean == pytest.approx(3.0)
    assert trend.std == pytest.approx(0.0)
    assert not trend.is_spike(3.0, z=3.0)  # rounding noise is not a spike


def test_the_mean_follows_a_drift_without_bias() -> None:
    """A plain EMA seeded at zero would lag for its whole memory; this one must not."""
    trend = Trend(beta_mean=0.9)
    for i in range(5):
        trend.update(float(i))
    assert trend.mean > 2.0  # a zero-seeded EMA would still sit near 1.6


def test_spread_reflects_the_deviations() -> None:
    trend = Trend()
    for i in range(200):
        trend.update(1.0 if i % 2 else 3.0)
    assert trend.mean == pytest.approx(2.0, abs=0.2)
    assert 0.5 < trend.std < 1.5


def test_spikes_are_measured_in_deviations_not_ratios() -> None:
    trend = Trend()
    for i in range(200):
        trend.update(1.0 if i % 2 else 3.0)
    threshold = trend.mean + 3.0 * trend.std

    assert not trend.is_spike(threshold - 1e-6, z=3.0)
    assert trend.is_spike(threshold + 1e-6, z=3.0)


def test_spike_detection_waits_for_warmup() -> None:
    trend = Trend(warmup=10)
    for _ in range(9):
        trend.update(1.0)
    assert not trend.is_spike(1e6, z=3.0)

    trend.update(1.0)
    assert trend.is_spike(1e6, z=3.0)


def test_a_disabled_threshold_never_spikes() -> None:
    trend = Trend()
    for _ in range(20):
        trend.update(1.0)
    assert not trend.is_spike(1e6, z=None)


def test_trend_rejects_bad_betas() -> None:
    with pytest.raises(ValueError, match="beta"):
        Trend(beta_mean=1.0)
    with pytest.raises(ValueError, match="beta"):
        Trend(beta_var=-0.1)


def test_trend_state_round_trip() -> None:
    trend = Trend()
    for i in range(30):
        trend.update(math.sin(i))

    restored = Trend()
    restored.load_state_dict(trend.state_dict())

    assert restored.mean == pytest.approx(trend.mean)
    assert restored.std == pytest.approx(trend.std)
    assert restored.count == trend.count


# -- EfficientRelativePollingSGD: the per-batch driver ---------------------------------

import torch  # noqa: E402
from torch import nn  # noqa: E402

from conftest import TinyNet, make_loader  # noqa: E402
from efficient_polling_lr_scheduler import (  # noqa: E402
    EfficientRelativePollingSGD,
    fit,
    make_closure,
)


class Scripted:
    """Drives an optimizer with scores and losses the test dictates.

    ``winner`` is the learning rate that should win the next poll; ``loss`` is
    what the closure reports at the current parameters. Both are writable
    mid-run, which is how the tests provoke a move, a spike or a blow-up.
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
        # Real gradients, but a reported loss value the test controls.
        raw = self.model(self.inputs).square().sum()
        loss = raw - raw.detach() + self.loss
        if not self.signal:
            return loss, 0.0
        return loss, 1.0 if math.isclose(lr, self.winner, rel_tol=1e-9) else 0.0


def clone_params(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def assert_params_equal(model: nn.Module, saved: list[torch.Tensor]) -> None:
    for param, value in zip(model.parameters(), saved, strict=True):
        assert torch.equal(param, value)


@pytest.fixture
def relative(batch):
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativePollingSGD(model, lr=1e-3, multiplier=10.0)
    return poller, Scripted(model, poller.optimizer, inputs, winner=1e-3), model


def test_first_batch_polls_three_candidates(relative) -> None:
    poller, closure, _ = relative
    info = poller.step(closure)
    assert info.polled
    assert info.optimizer_steps == 3 + 1  # three trials, then the winner reapplied


def test_the_window_recentres_on_the_winner(relative) -> None:
    poller, closure, _ = relative
    closure.winner = 1e-2  # X * m
    assert poller.step(closure).lr == pytest.approx(1e-2)

    closure.winner = 1e-1  # only a candidate if the window moved to 1e-2
    info = poller.step(closure)  # a changed choice polls again at once
    assert info.polled
    assert info.lr == pytest.approx(1e-1)


def test_ties_keep_the_current_rate_and_keep_polling(relative) -> None:
    poller, closure, _ = relative
    closure.signal = False  # every candidate ties, as at initialization
    for _ in range(30):
        info = poller.step(closure)
        assert info.polled
        assert info.lr == pytest.approx(1e-3)
    assert not poller.backoff.signal_seen


def test_interval_doubles_without_a_cap(relative) -> None:
    poller, closure, _ = relative
    intervals = []
    for _ in range(1100):
        info = poller.step(closure)
        if info.polled:
            intervals.append(info.poll_interval)
    assert intervals[:11] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def test_blind_steps_reuse_the_current_rate(relative) -> None:
    poller, closure, _ = relative
    poller.step(closure)  # a poll with signal that confirms the rate: interval 1
    info = poller.step(closure)
    assert not info.polled
    assert info.lr == pytest.approx(1e-3)
    assert info.optimizer_steps == 1


def test_a_spike_above_the_trend_forces_a_poll(relative) -> None:
    poller, closure, _ = relative
    for _ in range(80):  # backed off, trend warmed up with zero spread
        poller.step(closure)
    assert poller.backoff.interval > 1

    closure.loss = 1.5  # above the trend, below the blow-up bar of 2 * first loss
    info = poller.step(closure)
    assert info.spike
    assert info.polled
    assert not info.rolled_back


def test_a_spike_poll_that_confirms_the_rate_does_not_grow_the_interval(relative) -> None:
    """Only scheduled polls grow k, as TCP grows its window once per round trip.

    Spikes are frequent while the trend's spread is still immature; letting each
    of their confirmations double the interval sent k past 30,000 within the
    first 250 batches of a CIFAR-10 run.
    """
    poller, closure, _ = relative
    for _ in range(80):
        poller.step(closure)
    assert not poller.backoff.due()
    before = poller.backoff.interval

    closure.loss = 1.5
    info = poller.step(closure)
    assert info.spike
    assert info.lr == pytest.approx(1e-3)
    assert poller.backoff.interval == before


def test_spike_z_none_disables_the_spike_tier(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativePollingSGD(model, lr=1e-3, spike_z=None, rollback_loss=1e9)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3)
    for _ in range(80):
        poller.step(closure)
    assert not poller.backoff.due()

    closure.loss = 1e5
    info = poller.step(closure)
    assert not info.spike
    assert not info.polled
    assert not info.rolled_back


def test_a_blow_up_goes_back_in_time_to_the_best_point(relative) -> None:
    poller, closure, model = relative
    initial = clone_params(model)
    for _ in range(5):  # the loss never improves, so the initial point stays the best
        poller.step(closure)
    assert not torch.equal(next(model.parameters()), initial[0])

    closure.loss = math.inf
    info = poller.step(closure)

    assert info.rolled_back
    assert info.optimizer_steps == 0
    assert_params_equal(model, initial)
    assert poller.lr == pytest.approx(1e-3)


def test_the_best_point_follows_the_slow_trend(relative) -> None:
    poller, closure, model = relative
    closure.winner = 1e-2
    for _ in range(5):
        poller.step(closure)
    assert poller.best_lr == pytest.approx(1e-3)  # captured before the first move

    closure.loss = 0.5  # the trend now improves, poll after poll
    best_weights = None
    for _ in range(40):
        before = clone_params(model)
        previous = poller.best_trend
        info = poller.step(closure)
        if info.polled and poller.best_trend < previous:
            best_weights = before
    assert best_weights is not None
    assert poller.best_lr == pytest.approx(1e-2)

    closure.loss = math.inf
    assert poller.step(closure).rolled_back
    assert_params_equal(model, best_weights)
    assert poller.lr == pytest.approx(1e-2)


def test_the_poll_after_a_restart_breaks_ties_downward(relative) -> None:
    poller, closure, _ = relative
    closure.signal = False
    for _ in range(3):
        poller.step(closure)
    assert poller.lr == pytest.approx(1e-3)

    closure.loss = math.inf
    assert poller.step(closure).rolled_back
    closure.loss = 1.0

    info = poller.step(closure)
    assert info.polled
    assert info.lr == pytest.approx(1e-4)  # one notch down, and no further

    info = poller.step(closure)
    assert info.polled
    assert info.lr == pytest.approx(1e-4)


def test_a_restart_zeroes_the_interval_and_halves_the_ceiling(relative) -> None:
    poller, closure, _ = relative
    for _ in range(60):
        poller.step(closure)
    assert poller.backoff.interval == 32

    closure.loss = math.inf
    poller.step(closure)
    assert poller.backoff.interval == 0
    assert poller.backoff.ceiling == 16


def test_lr_max_folds_the_upper_candidate_into_the_bound(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativePollingSGD(model, lr=1e-2, lr_max=1e-2)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-2)

    info = poller.step(closure)
    assert info.optimizer_steps == 2 + 1  # {1e-3, 1e-2}: nothing above the bound


def test_lr_min_folds_the_lower_candidate_into_the_bound(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativePollingSGD(model, lr=1e-2, lr_min=1e-2)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-2)

    info = poller.step(closure)
    assert info.optimizer_steps == 2 + 1  # {1e-2, 1e-1}: nothing below the bound


def test_a_non_finite_loss_before_any_poll_is_survivable(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativePollingSGD(model, lr=1e-3)
    closure = Scripted(model, poller.optimizer, inputs, winner=1e-3, loss=math.inf)

    assert poller.step(closure).rolled_back
    closure.loss = 1.0
    assert poller.step(closure).polled


def test_rollback_threshold_is_derived_from_the_first_loss(relative) -> None:
    poller, closure, _ = relative
    closure.loss = math.log(10)
    poller.step(closure)
    assert poller.rollback_loss == pytest.approx(2.0 * math.log(10))


def test_state_dict_round_trip(relative) -> None:
    poller, closure, _ = relative
    closure.winner = 1e-2
    for _ in range(40):
        poller.step(closure)
    saved = poller.state_dict()

    restored = EfficientRelativePollingSGD(TinyNet(), lr=1e-3)
    restored.load_state_dict(saved)

    assert restored.backoff.state_dict() == poller.backoff.state_dict()
    assert restored.lr == pytest.approx(poller.lr)
    assert restored.best_lr == pytest.approx(poller.best_lr)
    assert restored.trend.mean == pytest.approx(poller.trend.mean)
    assert restored.slow_trend.mean == pytest.approx(poller.slow_trend.mean)
    assert restored.rollback_loss == pytest.approx(poller.rollback_loss)


def test_rejects_a_multiplier_not_above_one(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        EfficientRelativePollingSGD(model, lr=1e-3, multiplier=1.0)


def test_rejects_inverted_bounds(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="lr_min"):
        EfficientRelativePollingSGD(model, lr=1e-3, lr_min=1e-2, lr_max=1e-3)


def test_rejects_an_initial_rate_outside_the_bounds(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="lr"):
        EfficientRelativePollingSGD(model, lr=1e-3, lr_max=1e-4)


def test_sgd_and_polling_kwargs_are_routed(model: TinyNet) -> None:
    poller = EfficientRelativePollingSGD(model, lr=1e-3, momentum=0.9, multiplier=3.0)
    assert poller.optimizer.param_groups[0]["momentum"] == 0.9
    assert poller.multiplier == 3.0


def test_training_makes_progress_on_real_data(loss_fn) -> None:
    torch.manual_seed(3)
    model = TinyNet()
    poller = EfficientRelativePollingSGD(model, lr=1e-3)

    history = fit(
        model,
        make_loader(n_batches=8),
        make_loader(n_batches=2),
        poller,
        loss_fn,
        epochs=15,
        log_fn=None,
    )

    assert history.train_loss[-1] < history.train_loss[0]
    assert all(math.isfinite(x) for x in history.train_loss)


def test_polls_a_small_fraction_of_batches_once_there_is_signal(loss_fn) -> None:
    """The headline claim, in miniature: with signal, most batches go unpolled.

    On 8-sample batches every candidate ties on accuracy and the method polls
    every batch by design; this batch is large enough for the candidates to
    disagree, so the backoff engages -- and, having no cap, passes 64.
    """
    torch.manual_seed(0)
    inputs = torch.randn(256, 16)
    targets = torch.randint(0, 10, (256,))
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))
    poller = EfficientRelativePollingSGD(model, lr=1e-3, lr_max=1e-1)
    closure = make_closure(model, loss_fn, inputs, targets)

    infos = [poller.step(closure) for _ in range(1000)]

    assert poller.backoff.signal_seen
    assert sum(i.polled for i in infos) / len(infos) < 0.15
    assert max(i.poll_interval for i in infos) > 64
    assert not any(i.rolled_back for i in infos)


def test_selects_by_real_batch_accuracy(batch, loss_fn) -> None:
    inputs, targets = batch
    model = TinyNet()
    poller = EfficientRelativePollingSGD(model, lr=1e-3)
    closure = make_closure(model, loss_fn, inputs, targets)
    info = poller.step(closure)
    assert info.polled
    assert info.post_score is not None


# -- EfficientRelativeEpochPolling: the per-epoch driver -------------------------------

from efficient_polling_lr_scheduler import EfficientRelativeEpochPolling, EpochStats  # noqa: E402


class FakeTraining:
    """Stands in for ``train_epoch`` and ``evaluate`` with outcomes the test dictates.

    ``train`` moves every parameter by the learning rate in effect, so an
    epoch's end state is recognizable, and reports a mean training loss of
    ``loss``, halved for ``winner`` while ``signal`` is on -- the epoch's
    training loss is what a poll epoch selects on. ``validate`` reports
    ``level``, so the test can make validation improve or stall from epoch to
    epoch; validation is what the best point is judged on.
    """

    def __init__(self, model, optimizer, winner: float, batches: int = 4) -> None:
        self.model = model
        self.optimizer = optimizer
        self.winner = winner
        self.batches = batches
        self.loss = 1.0
        self.level = 0.0
        self.signal = True
        self.diverging: set[float] = set()
        self.calls: list[float] = []

    def _lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def train(self) -> EpochStats:
        lr = self._lr()
        self.calls.append(lr)
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(lr)
        if any(math.isclose(lr, d) for d in self.diverging):
            loss = math.nan
        elif self.signal and math.isclose(lr, self.winner, rel_tol=1e-9):
            loss = 0.5 * self.loss
        else:
            loss = self.loss
        return EpochStats(
            loss=loss, score=0.5, lr=lr, batches=self.batches, optimizer_steps=self.batches
        )

    def validate(self) -> tuple[float, float]:
        return 0.9, self.level


@pytest.fixture
def epoch_setup():
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    fake = FakeTraining(model, optimizer, winner=1e-3)
    controller = EfficientRelativeEpochPolling(multiplier=10.0)
    return controller, fake, model, optimizer


def run_epoch(controller, fake, model, optimizer):
    return controller.run_epoch(optimizer, model, train=fake.train, validate=fake.validate)


def test_a_poll_epoch_trains_every_candidate_and_keeps_the_winner(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    fake.winner = 1e-2
    initial = clone_params(model)

    stats, val_loss, val_score = run_epoch(controller, fake, model, optimizer)

    assert fake.calls == pytest.approx([1e-3, 1e-4, 1e-2])  # centre first: ties keep it
    assert stats.polls == fake.batches
    assert stats.optimizer_steps == 3 * fake.batches
    assert stats.lr == pytest.approx(1e-2)
    assert stats.loss == pytest.approx(0.5)  # the winner's epoch, selected on its loss
    assert val_score == pytest.approx(fake.level)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)
    assert_params_equal(model, [p + 1e-2 for p in initial])  # the winner's end state


def test_ties_keep_the_centre_and_poll_again(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    fake.signal = False
    for _ in range(3):
        stats, _, _ = run_epoch(controller, fake, model, optimizer)
        assert stats.polls == fake.batches
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert not controller.backoff.signal_seen


def test_blind_epochs_reuse_the_winner(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    run_epoch(controller, fake, model, optimizer)  # a confirming poll: interval 1
    fake.calls.clear()

    stats, _, _ = run_epoch(controller, fake, model, optimizer)

    assert fake.calls == pytest.approx([1e-3])
    assert stats.polls == 0
    assert stats.optimizer_steps == fake.batches
    assert controller.backoff.interval == 1


def test_a_blown_up_epoch_goes_back_to_the_best_epoch(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    best_weights = None
    for level in (0.1, 0.2, 0.3):  # validation improves, so each epoch is the new best
        fake.level = level
        run_epoch(controller, fake, model, optimizer)
        best_weights = clone_params(model)
    assert controller.best_score == pytest.approx(0.3)
    assert controller.backoff.interval >= 1
    fake.level = 0.0  # a worse epoch: not the best
    run_epoch(controller, fake, model, optimizer)

    fake.loss = math.nan
    stats, val_loss, val_score = run_epoch(controller, fake, model, optimizer)

    assert stats.rollbacks == 1
    assert_params_equal(model, best_weights)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert math.isfinite(val_loss)  # validated at the restored point
    assert controller.backoff.interval == 0
    assert controller.backoff.after_restart


def test_the_poll_epoch_after_a_restart_breaks_ties_downward(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    fake.signal = False
    run_epoch(controller, fake, model, optimizer)
    fake.loss = math.nan
    run_epoch(controller, fake, model, optimizer)
    fake.loss = 1.0
    fake.calls.clear()

    run_epoch(controller, fake, model, optimizer)

    assert fake.calls == pytest.approx([1e-4, 1e-3, 1e-2])  # ascending: ties go down
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    fake.calls.clear()
    run_epoch(controller, fake, model, optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)  # and no further


def test_a_diverging_trial_loses_the_poll(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    fake.winner = 1e-2
    fake.diverging = {1e-2}

    stats, _, val_score = run_epoch(controller, fake, model, optimizer)

    assert stats.rollbacks == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)  # ties keep the centre
    assert stats.optimizer_steps == 3 * fake.batches  # the lost trial still cost steps
    assert controller.backoff.signal_seen  # a divergence is a difference


def test_rollback_threshold_is_derived_from_the_first_epoch_loss(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    fake.signal = False  # every trial reports the same loss
    fake.loss = 0.7
    run_epoch(controller, fake, model, optimizer)
    assert controller.rollback_loss == pytest.approx(1.4)

    fake.loss = 1.5  # above the derived bar: a blow-up
    stats, _, _ = run_epoch(controller, fake, model, optimizer)
    assert stats.rollbacks == 1


def test_epoch_polling_rejects_a_bad_multiplier() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        EfficientRelativeEpochPolling(multiplier=0.5)


def test_fit_drives_epoch_polling_through_one_loop(loss_fn) -> None:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_loader = make_loader(n_batches=6)

    history = fit(
        model,
        train_loader,
        make_loader(n_batches=2),
        optimizer,
        loss_fn,
        epochs=3,
        epoch_polling=EfficientRelativeEpochPolling(),
        log_fn=None,
    )

    assert len(history.train_loss) == 3
    assert history.polls[0] == 6  # the first epoch is a poll epoch
    assert history.optimizer_steps[0] == 3 * 6
    assert all(math.isfinite(x) for x in history.val_loss)


def test_fit_refuses_epoch_polling_over_a_polling_optimizer(loss_fn) -> None:
    model = TinyNet()
    with pytest.raises(ValueError, match="epoch_polling"):
        fit(
            model,
            make_loader(n_batches=2),
            make_loader(n_batches=1),
            EfficientRelativePollingSGD(model, lr=1e-3),
            loss_fn,
            epochs=1,
            epoch_polling=EfficientRelativeEpochPolling(),
            log_fn=None,
        )


def test_epoch_polling_makes_progress_on_real_data(loss_fn) -> None:
    torch.manual_seed(3)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    history = fit(
        model,
        make_loader(n_batches=8),
        make_loader(n_batches=2),
        optimizer,
        loss_fn,
        epochs=12,
        epoch_polling=EfficientRelativeEpochPolling(lr_max=1.0),
        log_fn=None,
    )

    assert history.train_loss[-1] < history.train_loss[0]
    assert all(math.isfinite(x) for x in history.train_loss)


def test_ties_after_signal_leave_the_interval_alone() -> None:
    """A tie is not evidence of stability, and not evidence of change either.

    Reading ties as stability freezes the initial plateau: one noisy poll shows
    signal, every later tie doubles the interval, and the rate never moves.
    """
    backoff = Backoff()
    stable(backoff, 3)
    assert backoff.interval == 4
    tie(backoff, 5)
    assert backoff.interval == 4


def test_an_unscheduled_confirmation_leaves_the_interval_alone() -> None:
    backoff = Backoff()
    stable(backoff, 3)
    backoff.polled(changed=False, had_signal=True, scheduled=False)
    assert backoff.interval == 4
    backoff.polled(changed=True, had_signal=True, scheduled=False)
    assert backoff.interval == 0  # a change is a change, whoever asked


# -- Window: the candidates widen while the criterion is blind ------------------

from efficient_polling_lr_scheduler.efficient_relative import Window  # noqa: E402


def test_window_starts_one_multiplier_wide() -> None:
    window = Window(multiplier=10.0)
    assert window.candidates(1e-3, ascending=False) == pytest.approx((1e-3, 1e-4, 1e-2))
    assert window.candidates(1e-3, ascending=True) == pytest.approx((1e-4, 1e-3, 1e-2))


def test_a_blind_poll_widens_the_window_by_one_multiplier() -> None:
    """Every candidate tied: the next poll looks farther in both directions."""
    window = Window(multiplier=10.0)
    window.polled(had_signal=False)
    assert window.candidates(1e-3, ascending=False) == pytest.approx((1e-3, 1e-5, 1e-1))
    window.polled(had_signal=False)
    assert window.candidates(1e-3, ascending=False) == pytest.approx((1e-3, 1e-6, 1e0))


def test_a_poll_with_signal_narrows_the_window_back() -> None:
    window = Window(multiplier=10.0)
    window.polled(had_signal=False)
    window.polled(had_signal=False)
    window.polled(had_signal=True)
    assert window.candidates(1e-3, ascending=False) == pytest.approx((1e-3, 1e-4, 1e-2))


def test_the_window_stops_widening_at_max_reach() -> None:
    window = Window(multiplier=10.0, max_reach=2)
    for _ in range(5):
        window.polled(had_signal=False)
    assert window.candidates(1e-3, ascending=False) == pytest.approx((1e-3, 1e-5, 1e-1))


def test_the_window_folds_into_its_bounds() -> None:
    window = Window(multiplier=10.0, lr_min=1e-4, lr_max=1e-1)
    window.polled(had_signal=False)
    window.polled(had_signal=False)  # would reach 1e-6 and 1e0
    assert window.candidates(1e-3, ascending=False) == pytest.approx((1e-3, 1e-4, 1e-1))


def test_reset_narrows_the_window() -> None:
    window = Window(multiplier=10.0)
    window.polled(had_signal=False)
    window.reset()
    assert window.candidates(1e-3, ascending=False) == pytest.approx((1e-3, 1e-4, 1e-2))


def test_window_state_round_trip() -> None:
    window = Window(multiplier=10.0)
    window.polled(had_signal=False)
    restored = Window(multiplier=10.0)
    restored.load_state_dict(window.state_dict())
    assert restored.reach == window.reach


def test_a_blind_poll_lets_the_optimizer_see_farther(relative) -> None:
    """From 1e-3, a rate two multipliers up is out of reach until a poll comes back blind."""
    poller, closure, _ = relative
    closure.signal = False
    assert poller.step(closure).lr == pytest.approx(1e-3)  # blind poll

    closure.signal = True
    closure.winner = 1e-1
    info = poller.step(closure)
    assert info.polled
    assert info.lr == pytest.approx(1e-1)
    assert poller.window.reach == 1  # signal narrows the window again


def test_a_blind_poll_epoch_lets_the_controller_see_farther(epoch_setup) -> None:
    controller, fake, model, optimizer = epoch_setup
    fake.signal = False
    run_epoch(controller, fake, model, optimizer)
    fake.signal = True
    fake.winner = 1e-1
    fake.calls.clear()

    run_epoch(controller, fake, model, optimizer)

    assert fake.calls == pytest.approx([1e-3, 1e-5, 1e-1])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-1)


def test_a_restart_narrows_the_window(relative) -> None:
    poller, closure, _ = relative
    closure.signal = False
    poller.step(closure)
    poller.step(closure)
    assert poller.window.reach == 3
    closure.loss = math.inf
    poller.step(closure)
    assert poller.window.reach == 1


def test_a_poll_epoch_selects_on_training_progress_not_on_validation(epoch_setup) -> None:
    """Validation at the end of an epoch rewards standing still at a selected best point.

    A rate that barely moves the weights keeps last epoch's validation score,
    which was itself the best seen, so greedy selection on it ratchets the
    rate down until nothing moves. The epoch's mean training loss measures the
    progress the epoch made instead.
    """
    controller, fake, model, optimizer = epoch_setup
    fake.winner = 1e-2  # the lowest training loss ...
    original_validate = fake.validate

    def validate_prefers_small_steps():
        val_loss, level = original_validate()
        return val_loss, level + (1.0 if math.isclose(fake._lr(), 1e-4) else 0.0)

    stats, _, _ = controller.run_epoch(
        optimizer, model, train=fake.train, validate=validate_prefers_small_steps
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)  # ... wins regardless
