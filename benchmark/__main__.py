"""Run the comparison from the command line, the same code the notebook drives.

    python -m benchmark --data-dir /path/to/cifar-10-batches-py --seeds 42 43 44 45 46
    python -m benchmark --dataset mnist --data-dir /path/to/MNIST --methods efficient --epochs 20
    python -m benchmark --data-dir /path/to/cifar-10-batches-py --round adam:1 --seeds 42 43
    python -m benchmark --data-dir /path/to/cifar-10-batches-py --ceiling 1e-7 --seeds 42 43

Run from the repository root. Every (method, seed) is recorded in
``results/<dataset>/`` when it ends and skipped the next time, so a sweep can be
stopped and resumed; the table at the end is read from those records.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .datasets import DATASETS
from .methods import METHODS, Hyperparameters, rate_text
from .rounds import CEILING_METHODS, ROUND_METHODS, Round, ceiling_experiment, round_experiment
from .sweep import Experiment, results_table


def _ceiling(text: str) -> float | None:
    """A rate ceiling; ``inf`` means none."""
    value = float(text)
    return None if math.isinf(value) else value


def _round(text: str) -> Round:
    """A learning-rate round, written ``OPTIMIZER:RATE``."""
    optimizer, colon, rate = text.partition(":")
    try:
        if not colon:
            raise ValueError(f"expected OPTIMIZER:RATE, such as adam:1e-7, got {text!r}")
        return Round(optimizer, float(rate))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


# Every setting a flag can change: (flag, group in Hyperparameters, field, type, help).
# The defaults shown by --help are the dataclass defaults, the values the
# recorded runs used.
SETTINGS: tuple[tuple[str, str, str, Callable[[str], Any], str], ...] = (
    ("--epochs", "training", "epochs", int, "epochs per run"),
    ("--batch-size", "training", "batch_size", int, "batch size"),
    ("--val-fraction", "training", "val_fraction", float, "validation share of the train split"),
    ("--num-workers", "training", "num_workers", int, "data loader workers"),
    (
        "--scheduled-lr",
        "comparators",
        "scheduled_lr",
        float,
        "initial rate of the cosine, step and plateau schedules; a round starts them at its own",
    ),
    ("--step-size", "comparators", "step_size", int, "epochs between two step decays"),
    ("--step-gamma", "comparators", "step_gamma", float, "factor of each step decay"),
    ("--plateau-factor", "comparators", "plateau_factor", float, "ReduceLROnPlateau factor"),
    ("--plateau-patience", "comparators", "plateau_patience", int, "ReduceLROnPlateau patience"),
    ("--cosine-eta-min", "comparators", "cosine_eta_min", float, "cosine annealing floor"),
    ("--sps-max-lr", "comparators", "sps_max_lr", float, "cap on the Polyak step"),
    ("--armijo-lr-max", "comparators", "armijo_lr_max", float, "rate Armijo searches down from"),
    ("--armijo-alpha", "comparators", "armijo_alpha", float, "Armijo sufficient-decrease constant"),
    ("--armijo-beta", "comparators", "armijo_beta", float, "Armijo backtracking factor"),
    ("--armijo-max-iters", "comparators", "armijo_max_iters", int, "Armijo backtracking steps"),
    (
        "--max-poll-interval",
        "efficient",
        "max_poll_interval",
        int,
        "Efficient Polling: backoff cap, in blind steps between polls",
    ),
    (
        "--spike-factor",
        "efficient",
        "spike_factor",
        float,
        "Efficient Polling: poll at once above this multiple of the loss EMA",
    ),
    ("--loss-ema-beta", "efficient", "loss_ema_beta", float, "Efficient Polling: loss EMA decay"),
    (
        "--poll-rate",
        "ablation",
        "poll_rate",
        float,
        "poll rate both trigger ablations are calibrated to",
    ),
    (
        "--multiplier",
        "relative",
        "multiplier",
        float,
        "Efficient Relative Polling: spacing of the candidates {X/m, X, X*m}",
    ),
    (
        "--relative-lr-max",
        "relative",
        "lr_max",
        _ceiling,
        "Efficient Relative Polling: ceiling of the candidates; inf lets the window roam",
    ),
    (
        "--spike-z",
        "relative",
        "spike_z",
        float,
        "Efficient Relative Polling: deviations above the loss trend that force a poll",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--dataset", default="cifar10", choices=list(DATASETS))
    parser.add_argument(
        "--data-dir",
        required=True,
        help="directory holding the dataset's files; benchmark/datasets.py says "
        "which files each one expects and where to get them",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        metavar="METHOD",
        help="default: every method, or every method of the round or of the ceiling test",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument(
        "--lr",
        type=float,
        default=None,
        help="start every method at this rate instead of 1e-3; the runs are recorded "
        "as <method>_lr<rate>, apart from the main table (the initial-rate robustness runs)",
    )
    variant.add_argument(
        "--round",
        type=_round,
        default=None,
        metavar="OPTIMIZER:RATE",
        help="a learning-rate round, such as adam:1e-7: every method of the round steps with "
        "that optimizer from that rate, the schedules included, with the ablations calibrated, "
        "recorded in results/<dataset>/rounds/<optimizer>_lr<rate>/",
    )
    variant.add_argument(
        "--ceiling",
        type=float,
        default=None,
        metavar="RATE",
        help="SPS and Armijo on SGD under this ceiling, "
        "recorded in results/<dataset>/ceilings/sgd_lr<rate>/",
    )
    parser.add_argument("--results-dir", default=None, help="default: results/<dataset>")
    parser.add_argument("--models-dir", default=None, help="default: models/")
    parser.add_argument("--device", default=None, help="default: cuda when there is one")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-run runs that already have a record"
    )
    parser.add_argument(
        "--calibrate-ablations",
        action="store_true",
        help="calibrate both trigger ablations to the poll rate efficient measured on the "
        "first seed, as the recorded runs of every dataset but CIFAR-10 were",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=None,
        help="seed whose efficient run the ablations are calibrated to; default: the first "
        "of --seeds, which is what a sweep over every seed uses",
    )

    defaults = Hyperparameters()
    settings = parser.add_argument_group("hyperparameters")
    for flag, group, name, kind, text in SETTINGS:
        default = getattr(getattr(defaults, group), name)
        settings.add_argument(flag, type=kind, default=default, help=f"{text} (default: {default})")

    args = parser.parse_args(argv)
    allowed, study = METHODS, None
    if args.round is not None:
        allowed, study = ROUND_METHODS, "a round"
    elif args.ceiling is not None:
        if not args.ceiling > 0:
            parser.error(f"--ceiling must be positive, got {args.ceiling}")
        allowed, study = CEILING_METHODS, "the ceiling test"
    if args.methods is None:
        args.methods = list(allowed)
    elif study is not None:
        outside = [method for method in args.methods if method not in allowed]
        if outside:
            parser.error(
                f"{', '.join(outside)} not part of {study}, whose methods are {', '.join(allowed)}"
            )
    return args


def build_experiment(args: argparse.Namespace) -> Experiment:
    hyperparameters = Hyperparameters()
    for flag, group, name, _, _ in SETTINGS:
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        setattr(getattr(hyperparameters, group), name, value)
    experiment = Experiment(
        dataset=args.dataset,
        data_dir=Path(args.data_dir),
        seeds=tuple(args.seeds),
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        device=args.device,
        hyperparameters=hyperparameters,
        calibrate_ablations=args.calibrate_ablations,
        calibration_seed=args.calibration_seed,
    )
    if args.round is not None:
        return round_experiment(experiment, args.round)
    if args.ceiling is not None:
        return ceiling_experiment(experiment, args.ceiling)
    return experiment


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    experiment = build_experiment(args)
    # Seed by seed, so an interrupted sweep leaves whole seeds behind.
    for seed in experiment.seeds:
        for method in args.methods:
            experiment.sweep(method, seeds=[seed], lr=args.lr, overwrite=args.overwrite)
    print_summary(args, experiment)


def print_summary(args: argparse.Namespace, experiment: Experiment) -> None:
    """The table of the methods asked for, read from their records."""
    runs = experiment.load_runs()
    title = experiment.spec.name
    if args.round is not None:
        title += f", round {args.round.title}"
    elif args.ceiling is not None:
        title += f", SPS and Armijo under a {rate_text(args.ceiling)} ceiling"
    print(f"\n### {title}, mean ± stdev over the recorded seeds\n")
    runs = {m: runs[m] for m in args.methods if m in runs}
    print(results_table(runs, experiment.hyperparameters))


if __name__ == "__main__":
    main()
