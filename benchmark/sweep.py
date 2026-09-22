"""Run the configurations over seeds, record every run, and read the records back.

A finished run is written to ``results/<dataset>/<method>_seed<N>.json`` and read
back instead of retrained afterwards, so a sweep can stop and resume at any point.
Every table and figure is drawn from those files, never from a run held in memory,
so a figure cannot disagree with the table next to it.

A variant of an experiment, such as a learning-rate round, records into a folder
of its own under ``results/<dataset>/``, which the main table never reads.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from efficient_polling_lr_scheduler import evaluate, fit

from .datasets import ArrayDataset, DatasetSpec, build_split, compute_mean_std, spec_for
from .methods import (
    ABLATIONS,
    LABELS,
    Hyperparameters,
    Training,
    build_epoch_polling,
    build_optimizer,
    initial_lr,
    label,
    optimizer_name,
    rate_text,
    run_key,
)
from .models import build_model

__all__ = [
    "REPO_DIR",
    "Experiment",
    "load_initial_lr_runs",
    "load_runs",
    "record_label",
    "markdown_table",
    "results_table",
    "table_cells",
    "run_setting",
    "summarize",
]

REPO_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Experiment:
    """One dataset's comparison: where its data and records live, and how it runs.

    ``calibrate_ablations`` sets the two trigger ablations to the poll rate
    Efficient Polling measured on ``calibration_seed``, the first seed by
    default, read from its record when an ablation run starts. That is how the
    recorded runs of every dataset except CIFAR-10 were made; CIFAR-10's used
    the paper's 5%. One process per run (``benchmark.pool``) holds one seed, so
    it names the seed the whole sweep calibrates from instead of inheriting it.
    """

    dataset: str
    data_dir: Path | str
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    results_dir: Path | str | None = None  # default: results/<dataset>
    models_dir: Path | str | None = None  # default: models/
    device: str | None = None  # default: cuda when there is one
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    calibrate_ablations: bool = False
    calibration_seed: int | None = None  # default: the first seed

    def __post_init__(self) -> None:
        self.spec: DatasetSpec = spec_for(self.dataset)
        self.data_dir = Path(self.data_dir)
        self.seeds = tuple(self.seeds)
        self.results_dir = Path(self.results_dir or REPO_DIR / "results" / self.spec.key)
        self.models_dir = Path(self.models_dir or REPO_DIR / "models")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data --------------------------------------------------------------

    @cached_property
    def normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and standard deviation of the whole training split."""
        return compute_mean_std(build_split(self.spec.key, self.data_dir, train=True))

    @cached_property
    def train_set(self) -> ArrayDataset:
        """The training split, normalized, before the seed carves validation out of it."""
        mean, std = self.normalization
        return build_split(self.spec.key, self.data_dir, train=True, mean=mean, std=std)

    @cached_property
    def test_set(self) -> ArrayDataset:
        """The test split, normalized with the training statistics."""
        mean, std = self.normalization
        return build_split(self.spec.key, self.data_dir, train=False, mean=mean, std=std)

    def loaders(self, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Train, validation and test loaders for one seed.

        The train/val split is drawn from the seed, so the seeds vary the
        partition as well as the weight initialization: a difference that
        survives both is a difference in the method, not one lucky split.
        """
        training = self.hyperparameters.training
        full = self.train_set
        val_size = max(1, int(len(full) * training.val_fraction))
        train_set, val_set = random_split(
            full,
            [len(full) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        common: dict[str, Any] = {
            "batch_size": training.batch_size,
            "num_workers": training.num_workers,
            "pin_memory": str(self.device).startswith("cuda"),
        }
        return (
            DataLoader(train_set, shuffle=True, **common),
            DataLoader(val_set, shuffle=False, **common),
            DataLoader(self.test_set, shuffle=False, **common),
        )

    # --- where runs live -----------------------------------------------------

    def result_path(self, method: str, seed: int, lr: float | None = None) -> Path:
        return Path(self.results_dir) / f"{run_key(method, lr)}_seed{seed}.json"

    def checkpoint_path(self, method: str, seed: int, lr: float | None = None) -> Path:
        """Best-epoch weights, named after the dataset as well as the run, or a
        second dataset's sweep would silently restore the first one's model."""
        return Path(self.models_dir) / f"{self.spec.key}_best_{run_key(method, lr)}_seed{seed}.pt"

    # --- running -------------------------------------------------------------

    @property
    def ablation_seed(self) -> int:
        """The seed whose Efficient Polling run the trigger ablations are calibrated to."""
        return self.seeds[0] if self.calibration_seed is None else self.calibration_seed

    def measured_poll_rate(self, seed: int | None = None) -> float:
        """The share of batches Efficient Polling polled on a seed, the calibration one by
        default."""
        seed = self.ablation_seed if seed is None else seed
        path = self.result_path("efficient", seed)
        if not path.exists():
            raise FileNotFoundError(
                f"calibrating the ablations reads {path}; run efficient on seed {seed} first"
            )
        return float(json.loads(path.read_text())["poll_fraction"])

    def hyperparameters_for(
        self, method: str, epochs: int | None = None, lr: float | None = None
    ) -> Hyperparameters:
        """The settings one run uses: the experiment's, with this run's overrides."""
        hp = self.hyperparameters
        training = replace(
            hp.training,
            epochs=hp.training.epochs if epochs is None else epochs,
            lr=hp.training.lr if lr is None else lr,
        )
        ablation = hp.ablation
        if self.calibrate_ablations and method in ABLATIONS:
            ablation = replace(ablation, poll_rate=self.measured_poll_rate())
        return replace(hp, training=training, ablation=ablation)

    def run_one(
        self, method: str, seed: int, epochs: int | None = None, lr: float | None = None
    ) -> dict[str, Any]:
        """Train one method on one seed and return everything worth keeping.

        ``lr`` starts the run at another rate than the default, for the
        initial-rate robustness runs, and keys it apart (see ``run_key``).
        """
        hp = self.hyperparameters_for(method, epochs, lr)
        epochs = hp.training.epochs
        key = run_key(method, lr)
        train_loader, val_loader, test_loader = self.loaders(seed)

        # Re-seeded here, so every method of a seed starts from the same weights
        # and sees the same batch order.
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = build_model(self.spec).to(self.device)
        optimizer, scheduler = build_optimizer(method, model, self.spec, hp, seed)
        epoch_polling = build_epoch_polling(method, self.spec, hp)
        loss_fn = nn.CrossEntropyLoss()
        checkpoint = self.checkpoint_path(method, seed, lr)

        setting = f"{optimizer_name(hp.training.optimizer)} {rate_text(hp.training.lr)}"
        print(
            f"\n=== {self.spec.name} | {key} | {setting} | seed {seed} | {epochs} epochs ===",
            flush=True,
        )
        if method in ABLATIONS:
            print(f"trigger calibrated to a {hp.ablation.poll_rate:.2%} poll rate", flush=True)
        started = time.perf_counter()
        history = fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            epochs=epochs,
            device=self.device,
            checkpoint_path=checkpoint,
            scheduler=scheduler,
            epoch_polling=epoch_polling,
        )
        elapsed = time.perf_counter() - started

        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, self.device)

        batches = epochs * len(train_loader)
        result = {
            "method": key,
            "seed": seed,
            "epochs": epochs,
            "lr0": hp.training.lr,
            "optimizer": hp.training.optimizer,
            "best_val": history.best_val_acc,
            "best_epoch": history.best_epoch,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "s_per_epoch": elapsed / epochs,
            "poll_fraction": sum(history.polls) / batches if batches else 0.0,
            "steps": sum(history.optimizer_steps),
            "batches": batches,
            "rollbacks": sum(history.rollbacks),
            "spikes": sum(history.spikes),
            # Kept whole, so every figure can be redrawn from the record alone.
            "history": history.as_dict(),
        }
        print(
            f"{key} seed {seed}: best val {result['best_val']:.4f} | "
            f"test acc {test_acc:.4f} loss {test_loss:.4f} | "
            f"polled {result['poll_fraction']:.2%} | steps {result['steps']:,} | "
            f"{result['s_per_epoch']:.2f} s/epoch",
            flush=True,
        )
        return result

    def sweep(
        self,
        method: str,
        seeds: Iterable[int] | None = None,
        epochs: int | None = None,
        lr: float | None = None,
        overwrite: bool = False,
    ) -> list[dict[str, Any]]:
        """Run one method over the seeds, each run read from its record when it has one.

        A restarted kernel, or a second pass after adding a seed, retrains
        nothing that already finished. Delete a record, or pass
        ``overwrite=True``, to run it again.
        """
        runs = []
        for seed in self.seeds if seeds is None else seeds:
            path = self.result_path(method, seed, lr)
            if path.exists() and not overwrite:
                print(f"{run_key(method, lr)} seed {seed}: recorded ({path.name})")
                runs.append(json.loads(path.read_text()))
                continue
            result = self.run_one(method, seed, epochs, lr)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written aside and moved into place, so a run killed mid-write (a
            # preempted cluster job) leaves no half a record to be read back.
            partial = path.with_name(path.name + ".partial")
            partial.write_text(json.dumps(result, indent=2))
            partial.replace(path)
            runs.append(result)
        return runs

    def load_runs(self) -> dict[str, list[dict[str, Any]]]:
        return load_runs(Path(self.results_dir))

    def variant(
        self,
        name: str,
        hyperparameters: Hyperparameters,
        calibrate_ablations: bool | None = None,
    ) -> Experiment:
        """The same dataset and seeds under other settings, recorded apart.

        Records go to ``<results_dir>/<name>/`` and checkpoints to
        ``<models_dir>/<name>/``, so two variants running at once never restore
        each other's weights and the main table never reads a variant's runs.
        The data this experiment already loaded is shared, not read again.
        """
        other = replace(
            self,
            results_dir=Path(self.results_dir) / name,
            models_dir=Path(self.models_dir) / name,
            hyperparameters=hyperparameters,
            calibrate_ablations=(
                self.calibrate_ablations if calibrate_ablations is None else calibrate_ablations
            ),
        )
        for cached in ("normalization", "train_set", "test_set"):
            if cached in self.__dict__:
                other.__dict__[cached] = self.__dict__[cached]
        return other


# --- reading the records back ------------------------------------------------


def load_runs(results_dir: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Every recorded run of the thirteen configurations, by method, in table order.

    The initial-rate robustness runs are left out; see :func:`load_initial_lr_runs`.
    """
    found: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(results_dir).glob("*_seed*.json")):
        record = json.loads(path.read_text())
        found.setdefault(record["method"], []).append(record)
    return {method: found[method] for method in LABELS if method in found}


def load_initial_lr_runs(
    results_dir: Path | str, methods: tuple[str, ...] = ("efficient", "efficient_relative")
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    """Runs of ``methods`` keyed by ``(method, starting rate)``, the default start included."""
    found: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for path in sorted(Path(results_dir).glob("*_seed*.json")):
        record = json.loads(path.read_text())
        base = record["method"].split("_lr")[0]
        if base in methods:
            # Runs recorded before the robustness sweep carry no lr0: they all
            # started at the default rate.
            lr0 = float(record.get("lr0", Training.lr))
            found.setdefault((base, lr0), []).append(record)
    return dict(sorted(found.items()))


def run_setting(run: dict[str, Any]) -> tuple[str, float]:
    """The base optimizer and starting rate a record says it ran with.

    Records written before the learning-rate rounds carry neither field when
    they are old enough, and no optimizer in any case: they stepped with SGD.
    """
    return run.get("optimizer", "sgd"), float(run.get("lr0", Training.lr))


def record_label(method: str, run: dict[str, Any]) -> str:
    """How a table names a recorded method: after the optimizer and rate it ran with."""
    return label(method, *run_setting(run))


def summarize(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation; the deviation of a single run is zero.

    A run that diverged records a loss of NaN or inf, which ``statistics``
    refuses with an exception. The mean carries it instead, as NaN or inf, and
    so does the deviation, as NaN: a table with a diverged seed says so. Plain
    float addition, because ``fmean`` raises on inf and -inf together.
    """
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    if not all(math.isfinite(v) for v in values):
        return sum(values) / len(values), math.nan
    return statistics.fmean(values), statistics.stdev(values)


# The columns a results table gives every method, after its name.
TABLE_COLUMNS = ("Best Val", "Test Acc", "Test Loss", "Polled", "Steps", "s/Epoch", "Seeds")


def table_cells(runs: list[dict[str, Any]]) -> list[str]:
    """One method's :data:`TABLE_COLUMNS`, mean ± sample stdev over its runs.

    A method with no runs yet gets a dash in every column.
    """
    if not runs:
        return ["—"] * len(TABLE_COLUMNS)
    val_m, val_s = summarize([r["best_val"] for r in runs])
    acc_m, acc_s = summarize([r["test_acc"] for r in runs])
    loss_m, loss_s = summarize([r["test_loss"] for r in runs])
    sec_m, sec_s = summarize([r["s_per_epoch"] for r in runs])
    poll_m, _ = summarize([r["poll_fraction"] for r in runs])
    steps_m, _ = summarize([float(r["steps"]) for r in runs])
    return [
        f"{val_m:.2%} ± {val_s:.2%}",
        f"{acc_m:.2%} ± {acc_s:.2%}",
        f"{loss_m:.4f} ± {loss_s:.4f}",
        f"{poll_m:.2%}" if poll_m else "n/a",
        f"{steps_m:,.0f}",
        f"{sec_m:.2f} ± {sec_s:.2f}",
        str(len(runs)),
    ]


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)])


def results_table(
    runs_per_method: dict[str, list[dict[str, Any]]], hyperparameters: Hyperparameters
) -> str:
    """The paper's table as Markdown: every column is mean ± sample stdev over the seeds.

    Test metrics come from each run's own best-validation checkpoint, so the
    test set never selects anything. ``hyperparameters`` are the ones the runs
    were made with, the experiment's (``Experiment.hyperparameters``): the
    Initial LR column reads them, since a record keeps its rate but not where a
    schedule or a polling grid started from it.
    """
    rows = [
        [
            record_label(method, runs[0]).strip(),
            initial_lr(method, hyperparameters),
            *table_cells(runs),
        ]
        for method, runs in runs_per_method.items()
        if runs
    ]
    return markdown_table(["Method", "Initial LR", *TABLE_COLUMNS], rows)
