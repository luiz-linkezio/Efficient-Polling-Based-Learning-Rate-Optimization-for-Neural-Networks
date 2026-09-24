from __future__ import annotations

import math
import random

import pytest
import torch
from torch import nn

import efficient_polling_lr_scheduler as package
from conftest import TinyNet, make_loader
from efficient_polling_lr_scheduler import (
    EfficientRelativeNarrowingPollingOptimizer,
    EfficientRelativeNarrowingPollingSGD,
    EfficientRelativePollingSGD,
    fit,
    make_closure,
)
from efficient_polling_lr_scheduler.efficient_relative import Window
from efficient_polling_lr_scheduler.efficient_relative_narrowing import NarrowingWindow

# -- NarrowingWindow: the jump between the centre and its neighbours ------------

X = 1e-3


def kept(window: NarrowingWindow, centre: float = X, times: int = 1) -> None:
    """``times`` polls with signal where the centre won."""
    for _ in range(times):
        window.polled(True, centre=centre, winner=centre)


def moved(window: NarrowingWindow, direction: int, centre: float = X) -> None:
    """A poll with signal that moved the rate one jump up (+1) or down (-1)."""
    winner = centre * window.factor if direction > 0 else centre / window.factor
    window.polled(True, centre=centre, winner=winner)


def blind(window: NarrowingWindow, times: int = 1) -> None:
    for _ in range(times):
        window.polled(False, centre=X, winner=X)


def exponent(window: NarrowingWindow) -> float:
    """The jump in orders of magnitude of the multiplier."""
    return math.log(window.factor, window.multiplier)


def test_the_jump_starts_at_the_multiplier() -> None:
    window = NarrowingWindow(multiplier=10.0)
    assert window.factor == 10.0
    assert window.candidates(X, ascending=False) == (X, X / 10.0, X * 10.0)
    assert window.narrow_patience == window.widen_patience == 8


# -- the patience: a behaviour has to last before the jump moves ----------------


def test_one_centre_win_does_not_narrow() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=8)
    kept(window, times=7)
    assert window.factor == 10.0
    assert window.narrow_patience == 1

    kept(window)
    assert window.factor == pytest.approx(10.0**0.5)


def test_centre_wins_and_reversals_are_one_behaviour() -> None:
    """With {10, 100, 1000}, 100 always winning says what 10 <-> 1000 says."""
    window = NarrowingWindow(multiplier=10.0, patience=4)
    moved(window, +1)  # a first move: no direction to compare with yet
    assert window.narrow_patience == 4
    moved(window, -1)  # reversal
    kept(window)  # centre
    moved(window, +1)  # reversal of the last move, down
    assert window.narrow_patience == 1
    assert window.depth == 0
    kept(window)
    assert window.depth == 1


def test_a_break_slows_the_countdown_without_giving_it_back() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=4, break_discount=0.5)
    moved(window, +1)  # first move, direction up
    kept(window, times=2)
    assert window.narrow_patience == 2

    moved(window, +1)  # up again: a poll of the other behaviour
    assert window.narrow_patience == 1.5  # discounted by half, not refilled
    assert window.widen_patience == 2  # two halves from the centre wins, then a whole poll

    kept(window)
    assert window.depth == 0
    kept(window)
    assert window.depth == 1


def test_a_zero_discount_pauses_the_countdown_on_a_break() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=4, break_discount=0.0)
    moved(window, +1)
    kept(window, times=2)
    moved(window, +1)
    assert window.narrow_patience == 2


def test_a_lasting_run_widens_the_jump_back_but_never_past_the_multiplier() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=2)
    kept(window, times=4)
    assert window.depth == 2

    moved(window, +1)  # first move
    depths = []
    for _ in range(6):
        moved(window, +1)
        depths.append(window.depth)
    assert depths == [2, 1, 1, 0, 0, 0]
    assert window.factor == 10.0


def test_when_the_jump_moves_both_patiences_start_again() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=4, break_discount=0.5)
    kept(window, times=3)
    assert (window.narrow_patience, window.widen_patience) == (1, 2.5)
    kept(window)
    assert window.depth == 1
    assert (window.narrow_patience, window.widen_patience) == (4, 4)


def test_a_lasting_plateau_never_widens() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=4)
    depths = []
    for _ in range(60):
        kept(window)
        depths.append(window.depth)
    assert depths == sorted(depths)
    assert window.depth == 15  # one narrowing per 4 centre wins, none of them capped


def test_a_patience_that_runs_out_at_a_cap_still_refills_both() -> None:
    """At the cap a plateau keeps refilling the widening patience it discounts."""
    window = NarrowingWindow(multiplier=10.0, max_narrowings=3, patience=4, break_discount=0.5)
    kept(window, times=60)  # the cap, then 48 more centre wins
    assert window.depth == window.max_narrowings
    moved(window, +1)  # first move
    for _ in range(3):
        moved(window, +1)
    assert window.depth == window.max_narrowings  # 3 run polls < patience 4
    moved(window, +1)
    assert window.depth == window.max_narrowings - 1


@pytest.mark.parametrize("last", ["run", "bracket"])
def test_when_both_patiences_run_out_the_poll_decides(last: str) -> None:
    window = NarrowingWindow(multiplier=10.0, patience=3, break_discount=0.5)
    kept(window, times=3)
    assert window.depth == 1
    moved(window, +1)  # first move: sets the direction
    kept(window)  # p_n 2,   p_w 2.5
    moved(window, +1)  # p_n 1.5, p_w 1.5
    if last == "run":
        kept(window)  # p_n 0.5, p_w 1
        moved(window, +1)  # p_n 0,   p_w 0: a run poll
        assert window.depth == 0
    else:
        moved(window, +1)  # p_n 1,   p_w 0.5
        kept(window)  # p_n 0,   p_w 0: a bracket poll
        assert window.depth == 2


def test_a_discount_floats_cannot_hold_still_runs_out_on_time() -> None:
    """0.6 is not exact in binary; the countdown must still hit zero where it should."""
    window = NarrowingWindow(multiplier=10.0, patience=8, break_discount=0.6)
    kept(window, times=8)
    moved(window, +1)  # first move
    for _ in range(5):
        moved(window, +1)  # p_w 3, p_n 5
    for _ in range(5):
        kept(window)  # p_n 0, p_w 0 exactly: the bracket poll decides
    assert window.depth == 2


def test_the_narrowing_is_a_fraction_of_the_jump() -> None:
    window = NarrowingWindow(multiplier=10.0, narrowing=0.25, max_narrowings=5, patience=1)
    exponents = []
    for _ in range(3):
        kept(window)
        exponents.append(exponent(window))
    assert exponents == pytest.approx([0.75, 0.75**2, 0.75**3])


def test_nothing_caps_the_narrowings_by_default() -> None:
    window = NarrowingWindow(multiplier=10.0, narrowing=0.5, patience=1)
    assert window.max_narrowings is None
    kept(window, times=10)
    assert window.depth == 10
    assert window.factor == pytest.approx(10.0 ** (0.5**10))


def test_max_narrowings_caps_the_narrowings() -> None:
    window = NarrowingWindow(multiplier=10.0, narrowing=0.5, max_narrowings=3, patience=1)
    kept(window, times=10)
    assert window.depth == 3
    assert window.factor == pytest.approx(10.0**0.125)


@pytest.mark.parametrize("narrowing", [0.25, 0.5, 0.9])
def test_the_jump_stops_narrowing_before_the_neighbours_round_onto_the_centre(
    narrowing: float,
) -> None:
    window = NarrowingWindow(multiplier=10.0, narrowing=narrowing, patience=1)
    kept(window, times=500)
    deepest = window.depth
    assert 0 < deepest < 500
    kept(window)
    assert window.depth == deepest
    centre, lower, upper = window.candidates(X, ascending=False)
    assert lower < centre < upper
    finer = window.multiplier ** ((1.0 - narrowing) ** (deepest + 1))
    assert math.isclose(X * finer, X, rel_tol=1e-9)  # one more would round onto X


def test_the_first_narrowing_lands_on_the_middle() -> None:
    window = NarrowingWindow(multiplier=10.0, narrowing=0.5, patience=1)
    kept(window)
    centre, lower, upper = window.candidates(X, ascending=False)
    assert centre == X
    assert upper == pytest.approx(math.sqrt(X * X * 10.0))  # between X and X * 10
    assert lower == pytest.approx(math.sqrt(X * X / 10.0))


# -- ties, bounds, resets ----------------------------------------------------------


def test_ties_on_a_narrowed_jump_count_toward_widening() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=2)
    kept(window, times=2)
    assert window.depth == 1

    blind(window)
    assert window.depth == 1
    blind(window)
    assert window.depth == 0
    blind(window)  # at the multiplier a tie widens the window at once, as in 2.0.0
    assert window.factor == 100.0


def test_a_tie_at_the_multiplier_widens_the_window_at_once() -> None:
    window = NarrowingWindow(multiplier=10.0)
    blind(window)
    assert window.factor == 100.0
    assert window.widen_patience == 8


def test_a_widened_window_that_sees_goes_back_to_the_multiplier() -> None:
    window = NarrowingWindow(multiplier=10.0)
    blind(window, times=3)
    assert window.factor == 1e4

    kept(window)  # the centre won, but across four decades: no bracket to count
    assert window.factor == 10.0
    assert window.narrow_patience == 8


@pytest.mark.parametrize("bound", ["lr_max", "lr_min"])
def test_a_centre_on_a_bound_counts_for_neither_behaviour(bound: str) -> None:
    window = NarrowingWindow(multiplier=10.0, patience=2, **{bound: X})
    assert len(window.candidates(X, ascending=False)) == 2
    kept(window, times=6)
    assert window.factor == 10.0
    assert window.narrow_patience == window.widen_patience == 2


@pytest.mark.parametrize(
    ("bound", "centre"),
    [("lr_max", 0.09999999999999999), ("lr_min", 1.0000000000000002e-05)],
)
def test_a_centre_within_rounding_of_a_bound_is_on_it(bound: str, centre: float) -> None:
    """Fractional jumps reach a bound an ulp off; the exception must still hold."""
    value = 0.1 if bound == "lr_max" else 1e-5
    window = NarrowingWindow(multiplier=10.0, patience=1, **{bound: value})
    assert centre != value
    assert window.at_bound(centre)
    assert len(window.candidates(centre, ascending=False)) == 2
    kept(window, centre=centre, times=3)
    assert window.factor == 10.0


def test_a_neighbour_within_rounding_of_a_bound_lands_on_it() -> None:
    window = NarrowingWindow(multiplier=10.0, lr_max=0.1, max_reach=6)
    blind(window, times=5)  # reach 6
    assert 1e-7 * 10.0**6 != 0.1
    assert window.candidates(1e-7, ascending=False)[-1] == 0.1


def test_a_centre_inside_the_bounds_counts() -> None:
    window = NarrowingWindow(multiplier=10.0, lr_min=X / 2, lr_max=X * 2, patience=1)
    kept(window)
    assert window.depth == 1


def test_bounds_do_not_block_moves() -> None:
    window = NarrowingWindow(multiplier=10.0, lr_max=X, patience=1)
    moved(window, -1, centre=X)
    moved(window, +1, centre=X / 10.0)
    assert window.depth == 1  # the reversal still counts


def test_a_reset_goes_back_to_the_multiplier_with_fresh_patience() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=2)
    moved(window, +1)
    kept(window, times=3)
    assert window.depth == 1
    assert window.narrow_patience == 1
    window.reset()
    assert window.factor == 10.0
    assert window.last_move == 0
    assert window.narrow_patience == window.widen_patience == 2

    moved(window, -1)
    assert window.narrow_patience == 2  # after a reset this is a first move, not a reversal


def test_patience_one_reacts_to_every_poll() -> None:
    """With a patience of one poll the rule is the per-event one: every poll moves the jump."""
    window = NarrowingWindow(multiplier=10.0, patience=1)
    moved(window, +1)
    assert window.factor == 10.0
    moved(window, -1)
    assert window.factor == pytest.approx(10.0**0.5)
    moved(window, +1)
    assert window.factor == pytest.approx(10.0**0.25)
    moved(window, +1)
    assert window.factor == pytest.approx(10.0**0.5)
    blind(window)
    assert window.factor == 10.0


def test_zero_narrowing_is_the_plain_window() -> None:
    """Over any sequence of outcomes, ``narrowing=0`` gives Window's jumps exactly."""
    rng = random.Random(0)
    for patience in (1, 8):
        plain = Window(multiplier=10.0)
        narrowing = NarrowingWindow(multiplier=10.0, narrowing=0.0, patience=patience)
        for _ in range(500):
            signal = rng.random() < 0.6
            move = rng.choice((-1, 0, 1))
            winner = X * narrowing.factor**move
            plain.polled(signal)
            narrowing.polled(signal, centre=X, winner=winner)
            assert narrowing.factor == plain.factor


def test_without_the_outcome_the_window_only_reads_the_signal() -> None:
    window = NarrowingWindow(multiplier=10.0)
    window.polled(True)
    assert window.depth == 0
    assert window.narrow_patience == 8
    window.polled(False)
    assert window.reach == 2


def test_narrowing_window_state_round_trip() -> None:
    window = NarrowingWindow(multiplier=10.0, patience=4)
    moved(window, -1)
    kept(window, times=6)
    moved(window, -1)

    restored = NarrowingWindow(multiplier=10.0, patience=4)
    restored.load_state_dict(window.state_dict())
    assert restored.state_dict() == window.state_dict()
    assert restored.factor == window.factor
    assert restored.narrow_patience == window.narrow_patience != 4


def test_a_plain_window_state_loads_as_the_full_multiplier() -> None:
    restored = NarrowingWindow(multiplier=10.0, patience=3)
    kept(restored, times=5)
    assert restored.depth == 1
    assert restored.narrow_patience == 1
    restored.load_state_dict(Window(multiplier=10.0).state_dict())
    assert restored.factor == 10.0
    assert restored.last_move == 0
    assert restored.narrow_patience == restored.widen_patience == 3


def test_a_state_without_patiences_loads_with_fresh_ones() -> None:
    """A state saved before the patience rule keeps its jump and gets full patiences."""
    restored = NarrowingWindow(multiplier=10.0, patience=3)
    kept(restored, times=2)
    restored.load_state_dict({"reach": 1, "depth": 2, "last_move": -1})
    assert restored.depth == 2
    assert restored.last_move == -1
    assert restored.narrow_patience == restored.widen_patience == 3


@pytest.mark.parametrize("narrowing", [-0.1, 1.0, math.nan])
def test_rejects_a_narrowing_outside_zero_to_one(narrowing: float) -> None:
    with pytest.raises(ValueError, match="narrowing"):
        NarrowingWindow(narrowing=narrowing)


def test_rejects_a_negative_number_of_narrowings() -> None:
    with pytest.raises(ValueError, match="max_narrowings"):
        NarrowingWindow(max_narrowings=-1)


def test_a_cap_of_zero_never_narrows() -> None:
    window = NarrowingWindow(multiplier=10.0, max_narrowings=0, patience=1)
    kept(window, times=5)
    assert window.depth == 0
    assert window.factor == 10.0


def test_rejects_a_patience_below_one() -> None:
    with pytest.raises(ValueError, match="patience"):
        NarrowingWindow(patience=0)


@pytest.mark.parametrize("discount", [-0.1, 1.0, 1.5, math.nan])
def test_rejects_a_break_discount_outside_zero_to_one(discount: float) -> None:
    with pytest.raises(ValueError, match="break_discount"):
        NarrowingWindow(break_discount=discount)


# -- EfficientRelativeNarrowingPollingSGD: the per-batch driver -------------------


class Scripted:
    """Scores the candidate the test names as the winner, and nothing else.

    ``winner`` is a function of the poll's candidates, so a test can say
    "the centre" or "the upper neighbour" without knowing the jump.
    """

    def __init__(self, model, optimizer, inputs, pick) -> None:
        self.model = model
        self.optimizer = optimizer
        self.inputs = inputs
        self.pick = pick
        self.loss = 1.0
        self.target: float | None = None

    def __call__(self):
        lr = float(self.optimizer.param_groups[0]["lr"])
        raw = self.model(self.inputs).square().sum()
        loss = raw - raw.detach() + self.loss
        if self.target is None:
            return loss, 0.0
        return loss, 1.0 if math.isclose(lr, self.target, rel_tol=1e-12) else 0.0

    def aim(self, poller) -> None:
        """Point the closure at the candidate the next poll should choose."""
        centre = poller.lr
        factor = poller.window.factor
        self.target = self.pick(centre, factor)


def centre_wins(centre: float, factor: float) -> float:
    return centre


def upper_wins(centre: float, factor: float) -> float:
    return centre * factor


def lower_wins(centre: float, factor: float) -> float:
    return centre / factor


def scripted(batch, pick, **kwargs):
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativeNarrowingPollingSGD(model, lr=X, **kwargs)
    return poller, Scripted(model, poller.optimizer, inputs, pick), model


def polled_step(poller, closure):
    """Step until the next poll, and return its StepInfo."""
    for _ in range(10_000):
        closure.aim(poller)
        info = poller.step(closure)
        if info.polled:
            return info
    raise AssertionError("no poll in 10,000 steps")


def test_the_optimizer_waits_out_its_patience_before_narrowing(batch) -> None:
    poller, closure, _ = scripted(batch, centre_wins)
    for _ in range(7):
        polled_step(poller, closure)
    assert poller.window.factor == 10.0
    polled_step(poller, closure)
    assert poller.window.factor == pytest.approx(10.0**0.5)


def test_the_optimizer_narrows_when_the_centre_wins(batch) -> None:
    poller, closure, _ = scripted(batch, centre_wins, patience=1)
    polled_step(poller, closure)
    assert poller.window.factor == pytest.approx(10.0**0.5)

    closure.pick = upper_wins
    info = polled_step(poller, closure)
    assert info.lr == pytest.approx(X * 10.0**0.5)  # off the decade lattice
    assert info.optimizer_steps == 3 + 1


def test_the_optimizer_settles_between_two_decades(batch) -> None:
    """Up a decade, back down, and the next poll probes the middle."""
    poller, closure, _ = scripted(batch, upper_wins, patience=1)
    assert polled_step(poller, closure).lr == pytest.approx(1e-2)

    closure.pick = lower_wins
    assert polled_step(poller, closure).lr == pytest.approx(1e-3)

    closure.pick = upper_wins
    assert polled_step(poller, closure).lr == pytest.approx(10.0**-2.5)


def test_the_narrowing_leaves_the_backoff_alone(batch) -> None:
    """Narrowing on a kept centre is the same poll that grows the interval, untouched."""
    poller, closure, _ = scripted(batch, centre_wins, patience=1)
    reference = EfficientRelativePollingSGD(TinyNet(), lr=X)
    intervals, reference_intervals = [], []
    for _ in range(4):
        intervals.append(polled_step(poller, closure).poll_interval)
    assert poller.window.depth == 4  # the jump moved on every one of those polls
    for _ in range(4):
        reference.backoff.polled(changed=False, had_signal=True)
        reference_intervals.append(reference.backoff.interval)
    assert intervals == reference_intervals


def test_ties_widen_a_narrowed_jump_back(batch) -> None:
    poller, closure, _ = scripted(batch, centre_wins, patience=1)
    polled_step(poller, closure)
    polled_step(poller, closure)
    assert poller.window.depth == 2

    closure.pick = lambda centre, factor: None  # every candidate ties
    info = polled_step(poller, closure)
    assert not info.had_signal
    assert poller.window.depth == 1


def test_the_centre_on_the_ceiling_keeps_the_jump(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativeNarrowingPollingSGD(model, lr=1e-1, lr_max=1e-1, patience=1)
    closure = Scripted(model, poller.optimizer, inputs, centre_wins)
    for _ in range(3):
        info = polled_step(poller, closure)
        assert info.optimizer_steps == 2 + 1
    assert poller.window.factor == 10.0


def test_a_ceiling_reached_an_ulp_short_is_still_the_ceiling(batch) -> None:
    inputs, _ = batch
    model = TinyNet()
    poller = EfficientRelativeNarrowingPollingSGD(
        model, lr=0.09999999999999999, lr_max=0.1, patience=1
    )
    closure = Scripted(model, poller.optimizer, inputs, centre_wins)
    for _ in range(3):
        info = polled_step(poller, closure)
        assert info.optimizer_steps == 2 + 1  # no duplicate trial at the ceiling
    assert poller.window.factor == 10.0


def test_a_restart_goes_back_to_the_full_multiplier(batch) -> None:
    poller, closure, _ = scripted(batch, upper_wins, patience=2)
    polled_step(poller, closure)
    closure.pick = centre_wins
    polled_step(poller, closure)
    polled_step(poller, closure)
    polled_step(poller, closure)
    assert poller.window.depth == 1
    assert poller.window.last_move == 1
    assert poller.window.narrow_patience == 1

    closure.loss = math.inf
    assert poller.step(closure).rolled_back
    assert poller.window.factor == 10.0
    assert poller.window.last_move == 0
    assert poller.window.narrow_patience == 2


def test_state_dict_carries_the_jump(batch) -> None:
    poller, closure, _ = scripted(batch, centre_wins, patience=2)
    for _ in range(5):
        polled_step(poller, closure)
    assert poller.window.depth == 2
    assert poller.window.narrow_patience == 1

    restored = EfficientRelativeNarrowingPollingSGD(TinyNet(), lr=X, patience=2)
    restored.load_state_dict(poller.state_dict())
    assert restored.window.state_dict() == poller.window.state_dict()
    assert restored.window.factor == poller.window.factor
    assert restored.lr == pytest.approx(poller.lr)


def test_zero_narrowing_trains_exactly_like_efficient_relative_polling() -> None:
    """Same seed, same batches: every step, rate and weight identical."""

    def train(cls, **kwargs):
        torch.manual_seed(0)
        model = TinyNet()
        loader = make_loader(n_batches=16, batch_size=16)
        poller = cls(model, lr=1e-3, **kwargs)
        loss_fn = nn.CrossEntropyLoss()
        infos = []
        for _ in range(8):
            for inputs, targets in loader:
                infos.append(poller.step(make_closure(model, loss_fn, inputs, targets)))
        return infos, [p.detach().clone() for p in model.parameters()]

    infos, weights = train(EfficientRelativeNarrowingPollingSGD, narrowing=0.0)
    reference_infos, reference_weights = train(EfficientRelativePollingSGD)
    assert infos == reference_infos
    for a, b in zip(weights, reference_weights, strict=True):
        assert torch.equal(a, b)


def test_trains_under_fit() -> None:
    torch.manual_seed(0)
    model = TinyNet()
    loader = make_loader(n_batches=32, batch_size=16)
    poller = EfficientRelativeNarrowingPollingSGD(model, lr=1e-3, lr_max=1.0)
    history = fit(model, loader, loader, poller, nn.CrossEntropyLoss(), epochs=3, log_fn=None)

    assert len(history.train_loss) == 3
    assert all(math.isfinite(loss) for loss in history.train_loss)
    assert history.polls[0] > 0


def test_real_training_leaves_the_decade_lattice() -> None:
    """With a real criterion, some batch settles between two decades.

    No ceiling: with one at 1 this toy run parks on it, where a centre win counts
    for neither behaviour and the jump rightly stays at the multiplier.
    """
    torch.manual_seed(0)
    model = TinyNet()
    loader = make_loader(n_batches=32, batch_size=16)
    poller = EfficientRelativeNarrowingPollingSGD(model, lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rates = set()
    for _ in range(6):
        for inputs, targets in loader:
            rates.add(poller.step(make_closure(model, loss_fn, inputs, targets)).lr)
    assert any(abs(math.log10(lr) - round(math.log10(lr))) > 1e-6 for lr in rates)


def test_the_general_optimizer_wraps_any_optimizer() -> None:
    model = TinyNet()
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    poller = EfficientRelativeNarrowingPollingOptimizer(
        adam,
        module=model,
        multiplier=4.0,
        narrowing=0.25,
        max_narrowings=2,
        patience=5,
        break_discount=0.25,
        lr_max=1e-1,
    )
    assert poller.optimizer is adam
    assert poller.multiplier == 4.0
    assert poller.narrowing == 0.25
    assert poller.max_narrowings == 2
    assert poller.patience == 5
    assert poller.break_discount == 0.25
    assert poller.lr_max == 1e-1


def test_the_sgd_class_forwards_sgd_settings() -> None:
    poller = EfficientRelativeNarrowingPollingSGD(
        TinyNet(), lr=1e-2, momentum=0.9, narrowing=0.3, patience=3, break_discount=0.25
    )
    assert poller.optimizer.param_groups[0]["momentum"] == 0.9
    assert poller.narrowing == 0.3
    assert poller.patience == 3
    assert poller.break_discount == 0.25
    assert poller.max_narrowings is None
    assert "narrowing=0.3" in repr(poller)
    assert "patience=3" in repr(poller)
    assert "break_discount=0.25" in repr(poller)


def test_rejects_a_bad_narrowing(model: TinyNet) -> None:
    with pytest.raises(ValueError, match="narrowing"):
        EfficientRelativeNarrowingPollingSGD(model, lr=1e-3, narrowing=1.0)


def test_the_package_exports_it_as_3_0_0() -> None:
    assert package.__version__ == "3.0.0"
    assert "EfficientRelativeNarrowingPollingSGD" in package.__all__
    assert "EfficientRelativeNarrowingPollingOptimizer" in package.__all__
