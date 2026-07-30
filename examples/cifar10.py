"""Reproduce the paper's CIFAR-10 comparison from the command line.

Mirrors ``notebooks/cifar10.ipynb``: the same 5-layer CNN (557,898 parameters,
no batch norm or dropout, so the optimizer is the only source of adaptation),
vanilla SGD, batch size 64, base learning rate 1e-3, seed 42.

    python examples/cifar10.py --data-dir /path/to/cifar-10-batches-py

Get the data (the *CIFAR-10 python* version) from
https://www.cs.toronto.edu/~kriz/cifar.html:

    curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzf cifar-10-python.tar.gz

Requires the ``examples`` extra: ``pip install "efficient-polling-lr-scheduler[examples]"``.
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from efficient_polling_lr_scheduler import EfficientPollingSGD, PollingSGD, evaluate, fit

METHODS = ("baseline", "polling", "efficient")


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="bytes")
    data = obj[b"data"]  # (N, 3072)
    labels = np.array(obj.get(b"labels") or obj.get(b"fine_labels"), dtype=np.int64)
    return data.reshape(-1, 3, 32, 32), labels


class CIFAR10Dataset(Dataset):
    """The pickled CIFAR-10 batches, optionally normalized per channel."""

    def __init__(
        self,
        data_dir: Path,
        train: bool = True,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        files = (
            [data_dir / f"data_batch_{i}" for i in range(1, 6)]
            if train
            else [data_dir / "test_batch"]
        )
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                "missing CIFAR-10 batch files: "
                + ", ".join(missing)
                + "\nPoint --data-dir at the extracted cifar-10-batches-py directory."
            )
        arrays = [_load_cifar_batch(f) for f in files]
        self.data = np.concatenate([a for a, _ in arrays])
        self.labels = np.concatenate([b for _, b in arrays])
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.data[index].copy()).float().div_(255.0)
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std
        return image, torch.tensor(int(self.labels[index]))


def compute_mean_std(dataset: Dataset, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and standard deviation over the training set."""
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0

    for images, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images = images.double()
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += images.square().sum(dim=(0, 2, 3))
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / n_pixels
    std = (channel_sq_sum / n_pixels - mean.square()).clamp_min(0).sqrt()
    return mean.float().view(3, 1, 1), std.float().view(3, 1, 1)


class SimpleCIFAR10CNN(nn.Module):
    """5-layer CNN, 557,898 parameters, deliberately free of batch norm and dropout."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
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


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(args.data_dir)
    mean, std = compute_mean_std(CIFAR10Dataset(data_dir, train=True))
    print(f"channel mean {mean.squeeze().tolist()} std {std.squeeze().tolist()}")

    full_train = CIFAR10Dataset(data_dir, train=True, mean=mean, std=std)
    test_ds = CIFAR10Dataset(data_dir, train=False, mean=mean, std=std)

    val_size = max(1, int(len(full_train) * args.val_fraction))
    train_ds, val_ds = random_split(
        full_train,
        [len(full_train) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    common = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def build_optimizer(method: str, model: nn.Module, args: argparse.Namespace):
    if method == "baseline":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    if method == "polling":
        return PollingSGD(model, lr=args.lr)
    # The paper pins the rollback threshold at twice the random-guess loss for
    # ten classes; passing it explicitly keeps the run bit-comparable.
    return EfficientPollingSGD(
        model,
        lr=args.lr,
        max_poll_interval=args.max_poll_interval,
        spike_factor=args.spike_factor,
        loss_ema_beta=args.loss_ema_beta,
        rollback_loss=2.0 * math.log(10),
    )


def run(method: str, args: argparse.Namespace, loaders) -> dict[str, float]:
    train_loader, val_loader, test_loader = loaders
    device = torch.device(args.device)

    # Same seed for every method, so they differ only in learning-rate logic.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = SimpleCIFAR10CNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = build_optimizer(method, model, args)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint = Path(args.models_dir) / f"cifar10_best_{method}.pt"

    print(f"\n=== {method} | {n_params:,} parameters | {device} ===")
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
    )
    elapsed = time.perf_counter() - started

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

    total_polls = sum(history.polls)
    total_batches = args.epochs * len(train_loader)
    print(
        f"{method}: best val {history.best_val_acc:.4f} | "
        f"test loss {test_loss:.4f} acc {test_acc:.4f} | "
        f"{elapsed / args.epochs:.2f} s/epoch | "
        f"polls {total_polls}/{total_batches} | "
        f"steps {sum(history.optimizer_steps)} | rollbacks {sum(history.rollbacks)}"
    )

    return {
        "best_val": history.best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "s_per_epoch": elapsed / args.epochs,
        "poll_fraction": total_polls / total_batches if total_batches else 0.0,
        "steps": sum(history.optimizer_steps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="extracted cifar-10-batches-py directory")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS), metavar="METHOD"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--max-poll-interval", type=int, default=64)
    parser.add_argument("--spike-factor", type=float, default=3.0)
    parser.add_argument("--loss-ema-beta", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaders = build_loaders(args)
    results = {method: run(method, args, loaders) for method in args.methods}

    print("\n| Method | Best Val | Test Acc | Test Loss | Polled | s/Epoch |")
    print("|---|---|---|---|---|---|")
    for method, r in results.items():
        polled = "—" if method == "baseline" else f"{r['poll_fraction']:.2%}"
        print(
            f"| {method} | {r['best_val']:.2%} | {r['test_acc']:.2%} | "
            f"{r['test_loss']:.4f} | {polled} | {r['s_per_epoch']:.2f} |"
        )


if __name__ == "__main__":
    main()
