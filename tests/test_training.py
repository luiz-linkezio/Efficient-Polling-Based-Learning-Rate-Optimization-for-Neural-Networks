from __future__ import annotations

import math

import pytest
import torch

from conftest import TinyNet, make_loader
from efficient_polling_lr_scheduler import (
    EfficientPollingSGD,
    PollingSGD,
    evaluate,
    fit,
    negative_loss,
    train_epoch,
)


def test_train_epoch_with_a_plain_optimizer(loss_fn) -> None:
    model = TinyNet()
    loader = make_loader(n_batches=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    stats = train_epoch(model, loader, optimizer, loss_fn)

    assert stats.batches == 5
    assert stats.optimizer_steps == 5
    assert stats.polls == 0
    assert stats.lr == pytest.approx(1e-2)
    assert math.isfinite(stats.loss)


def test_train_epoch_with_base_polling(loss_fn) -> None:
    model = TinyNet()
    loader = make_loader(n_batches=5)
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=(1e-4, 1e-3, 1e-2))

    stats = train_epoch(model, loader, poller, loss_fn)

    assert stats.polls == 5  # every batch
    assert stats.optimizer_steps == 5 * (3 + 1)


def test_train_epoch_with_efficient_polling_lr_scheduler(loss_fn) -> None:
    model = TinyNet()
    loader = make_loader(n_batches=40)
    poller = EfficientPollingSGD(model, lr=1e-3)

    stats = train_epoch(model, loader, poller, loss_fn)

    assert stats.batches == 40
    assert 0 < stats.polls < 40  # cheaper than the base method
    assert stats.optimizer_steps < 40 * (len(poller.candidate_lrs) + 1)


def test_evaluate_reports_loss_and_accuracy(loss_fn) -> None:
    model = TinyNet()
    loader = make_loader(n_batches=3)

    loss, acc = evaluate(model, loader, loss_fn)

    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0


def test_evaluate_restores_training_mode(loss_fn) -> None:
    model = TinyNet()
    model.train()
    evaluate(model, make_loader(n_batches=2), loss_fn)
    assert model.training


def test_fit_records_one_entry_per_epoch(loss_fn) -> None:
    model = TinyNet()
    train_loader = make_loader(n_batches=6)
    val_loader = make_loader(n_batches=2)
    poller = EfficientPollingSGD(model, lr=1e-3)

    history = fit(model, train_loader, val_loader, poller, loss_fn, epochs=3, log_fn=None)

    assert len(history.train_loss) == 3
    assert len(history.val_acc) == 3
    assert len(history.polls) == 3
    assert 1 <= history.best_epoch <= 3
    assert set(history.as_dict()) >= {"train_loss", "val_acc", "lr", "polls"}


def test_fit_writes_the_best_checkpoint(loss_fn, tmp_path) -> None:
    model = TinyNet()
    path = tmp_path / "nested" / "best.pt"
    poller = PollingSGD(model, lr=1e-2, candidate_lrs=(1e-3, 1e-2))

    fit(
        model,
        make_loader(n_batches=4),
        make_loader(n_batches=2),
        poller,
        loss_fn,
        epochs=2,
        checkpoint_path=path,
        log_fn=None,
    )

    assert path.exists()
    assert set(torch.load(path, weights_only=True)) == {"linear.weight", "linear.bias"}


def test_fit_runs_the_baseline_too(loss_fn) -> None:
    """The same helper has to serve the fixed-LR baseline for comparison."""
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    history = fit(
        model,
        make_loader(n_batches=4),
        make_loader(n_batches=2),
        optimizer,
        loss_fn,
        epochs=2,
        log_fn=None,
    )

    assert history.optimizer_steps == [4, 4]


def test_selection_by_loss(loss_fn) -> None:
    model = TinyNet()
    loader = make_loader(n_batches=4)
    poller = PollingSGD(model, lr=1e-3, candidate_lrs=(1e-4, 1e-3, 1e-2))

    stats = train_epoch(model, loader, poller, loss_fn, score_fn=negative_loss)

    assert stats.polls == 4
    assert stats.score <= 0.0  # scores are negated losses


def test_training_makes_progress(loss_fn) -> None:
    torch.manual_seed(3)
    model = TinyNet()
    train_loader = make_loader(n_batches=8)
    val_loader = make_loader(n_batches=2)
    poller = EfficientPollingSGD(model, lr=1e-3)

    history = fit(model, train_loader, val_loader, poller, loss_fn, epochs=15, log_fn=None)

    assert history.train_loss[-1] < history.train_loss[0]
    assert sum(history.rollbacks) == 0
