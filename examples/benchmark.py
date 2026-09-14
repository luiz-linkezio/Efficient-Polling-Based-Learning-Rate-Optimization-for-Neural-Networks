"""Reproduce the paper's comparison from the command line, on any of four datasets.

The single source of truth for every number in the paper: all methods run
through the same :func:`fit` loop, on the same measurement convention (batch
loss and accuracy are recorded at the point where the gradient was taken,
before the step) and the same cost accounting, so the accuracy, optimizer-step
and wall-clock columns of a table always come from one implementation.

The same 5-layer CNN (no batch norm or dropout, so the optimizer is the only
source of adaptation), vanilla SGD, batch size 64, base learning rate 1e-3.
Only the first convolution and the classifier change with the dataset, which
leaves 557,898 parameters on CIFAR-10, the width the paper quotes. A dataset
with no spatial axes gets the dense network instead, built on the same rule:
nothing in it adapts except the optimizer.

    python examples/benchmark.py --data-dir /path/to/cifar-10-batches-py \\
        --seeds 42 43 44 45 46
    python examples/benchmark.py --dataset mnist --data-dir /path/to/MNIST

``--dataset`` picks between ``cifar10`` (the default, the paper's benchmark),
``cifar100``, ``mnist``, ``fashion_mnist`` and ``covertype``, the tabular one;
see ``examples/benchmark_datasets.py`` for where to get each. Results default to
``results/<dataset>/``, so the datasets never overwrite one another.

Each (method, seed) run is written to ``--results-dir`` as JSON and skipped on
a later invocation, so a long sweep can be interrupted and resumed. Results are
reported as mean +- sample standard deviation over the seeds.

Requires the ``examples`` extra: ``pip install "efficient-polling-lr-scheduler[examples]"``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark_datasets import (
    DATASETS,
    DatasetSpec,
    blowup_loss,
    build_split,
    compute_mean_std,
    spec_for,
)
from torch import nn
from torch.utils.data import DataLoader, random_split

from efficient_polling_lr_scheduler import (
    SPSSGD,
    ArmijoSGD,
    EfficientPollingSGD,
    EfficientRelativeEpochPolling,
    EfficientRelativePollingSGD,
    PollingSGD,
    evaluate,
    fit,
)

# Ordered so the three methods the paper's claim rests on run first: an
# interrupted sweep still leaves the core comparison complete.
METHODS = (
    "baseline",
    "polling",
    "efficient",
    "adam",
    "cosine",
    "step",
    "plateau",
    "sps",
    "armijo",
    "efficient_fixed",
    "efficient_random",
    "efficient_relative",
    "efficient_relative_epoch",
)

LABELS = {
    "baseline": "SGD (fixed 1e-3)",
    "polling": "Polling (base paper)",
    "efficient": "Efficient Polling (ours)",
    "adam": "Adam (1e-3)",
    "cosine": "SGD + cosine annealing",
    "step": "SGD + step decay",
    "plateau": "SGD + ReduceLROnPlateau",
    "sps": "SPS (Polyak)",
    "armijo": "Armijo line search",
    "efficient_fixed": "  ablation: fixed interval",
    "efficient_random": "  ablation: random trigger",
    "efficient_relative": "Efficient Relative Polling (ours, per batch)",
    "efficient_relative_epoch": "Efficient Relative Polling (ours, per epoch)",
}


class SimpleCNN(nn.Module):
    """5-layer CNN, deliberately free of batch norm and dropout.

    The global pool before the classifier means the input side length never
    enters: 28x28 and 32x32 both arrive at the same 256 features, so only the
    channel count and the class count vary between datasets.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleMLP(nn.Module):
    """The dense counterpart of :class:`SimpleCNN`, for data with no spatial
    axes to convolve over. Same rule: no batch norm and no dropout, so the
    optimizer stays the only source of adaptation."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        widths: tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_features
        for width in widths:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(spec: DatasetSpec) -> nn.Module:
    """The network the dataset's shape calls for. The learning-rate logic is the
    object of study, so the architecture is the plainest one that fits."""
    if spec.is_image:
        return SimpleCNN(in_channels=spec.in_channels, num_classes=spec.num_classes)
    (features,) = spec.input_shape
    return SimpleMLP(in_features=features, num_classes=spec.num_classes)


def build_loaders(
    args: argparse.Namespace, spec: DatasetSpec, seed: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders for one seed. The train/val split follows the seed, so the seeds
    vary the split as well as the initialization."""
    data_dir = Path(args.data_dir)
    mean, std = compute_mean_std(build_split(spec.key, data_dir, train=True))

    full_train = build_split(spec.key, data_dir, train=True, mean=mean, std=std)
    test_ds = build_split(spec.key, data_dir, train=False, mean=mean, std=std)

    val_size = max(1, int(len(full_train) * args.val_fraction))
    train_ds, val_ds = random_split(
        full_train,
        [len(full_train) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    common = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def build_optimizer(
    method: str, model: nn.Module, args: argparse.Namespace, seed: int, spec: DatasetSpec
):
    """Returns ``(optimizer, scheduler)``; ``scheduler`` is ``None`` for most methods."""
    if method == "baseline":
        return torch.optim.SGD(model.parameters(), lr=args.lr), None
    if method == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr), None
    if method == "cosine":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.scheduled_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        return optimizer, scheduler
    if method == "step":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.scheduled_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )
        return optimizer, scheduler
    if method == "plateau":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.scheduled_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return optimizer, scheduler
    if method == "sps":
        return SPSSGD(model, lr=args.lr, max_lr=args.sps_max_lr), None
    if method == "armijo":
        return (
            ArmijoSGD(
                model,
                lr=args.lr,
                lr_max=args.armijo_lr_max,
                alpha=args.armijo_alpha,
                beta=args.armijo_beta,
                max_iters=args.armijo_max_iters,
            ),
            None,
        )
    if method == "polling":
        return PollingSGD(model, lr=args.lr), None
    if method == "efficient_relative":
        return (
            EfficientRelativePollingSGD(
                model,
                lr=args.lr,
                multiplier=args.multiplier,
                lr_max=_relative_lr_max(args),
                spike_z=args.spike_z,
                rollback_loss=blowup_loss(spec),
            ),
            None,
        )
    if method == "efficient_relative_epoch":
        # Per epoch the controller drives a plain SGD; see build_epoch_polling().
        return torch.optim.SGD(model.parameters(), lr=args.lr), None
    # The paper pins the rollback threshold at twice the random-guess loss;
    # passing it explicitly keeps the run bit-comparable, and deriving it from
    # the class count keeps it meaningful when the dataset changes.
    #
    # The two ablations change the trigger and nothing else, both calibrated to
    # --poll-rate so they poll as often as the backoff variant does.
    kwargs: dict[str, Any] = {
        "spike_factor": args.spike_factor,
        "loss_ema_beta": args.loss_ema_beta,
        "rollback_loss": blowup_loss(spec),
    }
    if method == "efficient":
        kwargs["max_poll_interval"] = args.max_poll_interval
    elif method == "efficient_fixed":
        kwargs.update(trigger="fixed", max_poll_interval=round(1 / args.poll_rate) - 1)
    elif method == "efficient_random":
        # Seeded per run, and private, so the trigger never shifts batch order.
        kwargs.update(trigger="random", poll_probability=args.poll_rate, poll_seed=seed)
    else:
        raise ValueError(f"unknown method: {method}")
    return EfficientPollingSGD(model, lr=args.lr, **kwargs), None


def _relative_lr_max(args: argparse.Namespace) -> float | None:
    return None if math.isinf(args.relative_lr_max) else args.relative_lr_max


def build_epoch_polling(method: str, args: argparse.Namespace, spec: DatasetSpec):
    """The epoch-level controller for ``efficient_relative_epoch``; ``None`` otherwise."""
    if method != "efficient_relative_epoch":
        return None
    return EfficientRelativeEpochPolling(
        multiplier=args.multiplier,
        lr_max=_relative_lr_max(args),
        rollback_loss=blowup_loss(spec),
    )


def run(method: str, seed: int, args: argparse.Namespace, spec: DatasetSpec, loaders) -> dict:
    train_loader, val_loader, test_loader = loaders
    device = torch.device(args.device)

    # Same seed for every method, so they differ only in learning-rate logic.
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(spec).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer, scheduler = build_optimizer(method, model, args, seed, spec)
    epoch_polling = build_epoch_polling(method, args, spec)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint = checkpoint_path(args, spec, method, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== {spec.name} | {method} | seed {seed} | {n_params:,} parameters | {device} ===",
        flush=True,
    )
    started = time.perf_counter()
    history = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        epochs=args.epochs,
        device=device,
        checkpoint_path=checkpoint,
        scheduler=scheduler,
        epoch_polling=epoch_polling,
    )
    elapsed = time.perf_counter() - started

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

    total_polls = sum(history.polls)
    total_batches = args.epochs * len(train_loader)
    print(
        f"{method} seed {seed}: best val {history.best_val_acc:.4f} | "
        f"test loss {test_loss:.4f} acc {test_acc:.4f} | "
        f"{elapsed / args.epochs:.2f} s/epoch | "
        f"polls {total_polls}/{total_batches} | "
        f"steps {sum(history.optimizer_steps)} | rollbacks {sum(history.rollbacks)}",
        flush=True,
    )

    return {
        "method": method,
        "dataset": spec.key,
        "seed": seed,
        "epochs": args.epochs,
        "lr0": args.lr,
        "best_val": history.best_val_acc,
        "best_epoch": history.best_epoch,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "s_per_epoch": elapsed / args.epochs,
        "total_seconds": elapsed,
        "poll_fraction": total_polls / total_batches if total_batches else 0.0,
        "steps": sum(history.optimizer_steps),
        "batches": total_batches,
        "rollbacks": sum(history.rollbacks),
        "spikes": sum(history.spikes),
        # Kept so the figures can be redrawn from the recorded runs alone.
        "history": history.as_dict(),
    }


def result_path(args: argparse.Namespace, method: str, seed: int) -> Path:
    return Path(args.results_dir) / f"{method}_seed{seed}.json"


def checkpoint_path(args: argparse.Namespace, spec: DatasetSpec, method: str, seed: int) -> Path:
    """Best-epoch weights. Named after the dataset as well as the method, or a
    second dataset's sweep would silently restore the first one's model."""
    return Path(args.models_dir) / f"{spec.key}_best_{method}_seed{seed}.pt"


def summarize(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (zero for a single run)."""
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=list(DATASETS),
        help="dataset to run on (default: the paper's CIFAR-10)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="directory holding that dataset's files; examples/benchmark_datasets.py "
        "says which files each one expects and where to get them",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS), metavar="METHOD"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="one run per seed per method; results are reported as mean +- stdev",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="where the per-run JSON goes (default: results/<dataset>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run (method, seed) pairs that already have a result file",
    )
    parser.add_argument("--max-poll-interval", type=int, default=64)
    parser.add_argument("--spike-factor", type=float, default=3.0)
    parser.add_argument("--loss-ema-beta", type=float, default=0.9)
    parser.add_argument(
        "--poll-rate",
        type=float,
        default=0.05,
        help="poll rate the two trigger ablations are calibrated to, so they poll "
        "as often as the backoff variant measured and the comparison is about "
        "the trigger rather than the budget",
    )
    parser.add_argument(
        "--scheduled-lr",
        type=float,
        default=1e-1,
        help="initial LR for the cosine, step and plateau schedules, whose whole "
        "point is to start high and decay (the top of the polling candidate set)",
    )
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--step-gamma", type=float, default=0.1)
    parser.add_argument(
        "--sps-max-lr",
        type=float,
        default=0.1,
        help="cap on the Polyak step. Set to the top of the polling candidate set, "
        "the same upper bound Armijo searches from, so no adaptive method may "
        "take a step the others were never allowed to consider. Uncapped (the "
        "literature's 10.0) the rule saturates from the first batch on this model",
    )
    parser.add_argument("--armijo-lr-max", type=float, default=0.1)
    parser.add_argument("--armijo-alpha", type=float, default=1e-4)
    parser.add_argument("--armijo-beta", type=float, default=0.5)
    parser.add_argument("--armijo-max-iters", type=int, default=10)
    parser.add_argument(
        "--multiplier",
        type=float,
        default=10.0,
        help="Efficient Relative Polling: spacing between the three candidates {X/m, X, X*m}",
    )
    parser.add_argument(
        "--spike-z",
        type=float,
        default=3.0,
        help="Efficient Relative Polling, per batch: deviations above the trend that force a poll",
    )
    parser.add_argument(
        "--relative-lr-max",
        type=float,
        default=0.1,
        help="ceiling for Efficient Relative Polling's candidates, the same 1e-1 every other "
        "method is held to; pass inf to let the window roam. For the initial-rate "
        "robustness runs, combine --lr with a separate --results-dir",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)
    if args.results_dir is None:
        args.results_dir = str(Path("results") / args.dataset)
    return args


def main() -> None:
    args = parse_args()
    spec = spec_for(args.dataset)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        pending = [
            m for m in args.methods if args.overwrite or not result_path(args, m, seed).exists()
        ]
        if not pending:
            print(f"seed {seed}: every method already has a result, skipping")
            continue

        # One set of loaders per seed, shared by every method of that seed.
        loaders = build_loaders(args, spec, seed)
        for method in pending:
            result = run(method, seed, args, spec, loaders)
            result_path(args, method, seed).write_text(json.dumps(result, indent=2))

    report(args, spec)


def report(args: argparse.Namespace, spec: DatasetSpec) -> None:
    """Aggregate whatever result files exist into the paper's table."""
    print(f"\n### {spec.name}, {args.epochs} epochs, mean +- stdev over seeds\n")
    print("| Method | Best Val | Test Acc | Test Loss | Polled | Steps | s/Epoch | Seeds |")
    print("|---|---|---|---|---|---|---|---|")

    for method in args.methods:
        runs = [
            json.loads(result_path(args, method, seed).read_text())
            for seed in args.seeds
            if result_path(args, method, seed).exists()
        ]
        if not runs:
            continue

        val_m, val_s = summarize([r["best_val"] for r in runs])
        acc_m, acc_s = summarize([r["test_acc"] for r in runs])
        loss_m, loss_s = summarize([r["test_loss"] for r in runs])
        sec_m, sec_s = summarize([r["s_per_epoch"] for r in runs])
        poll_m, _ = summarize([r["poll_fraction"] for r in runs])
        steps_m, _ = summarize([float(r["steps"]) for r in runs])

        polled = f"{poll_m:.2%}" if poll_m else "—"
        print(
            f"| {LABELS.get(method, method)} "
            f"| {val_m:.2%} ± {val_s:.2%} "
            f"| {acc_m:.2%} ± {acc_s:.2%} "
            f"| {loss_m:.4f} ± {loss_s:.4f} "
            f"| {polled} | {steps_m:,.0f} "
            f"| {sec_m:.2f} ± {sec_s:.2f} | {len(runs)} |"
        )


if __name__ == "__main__":
    main()
