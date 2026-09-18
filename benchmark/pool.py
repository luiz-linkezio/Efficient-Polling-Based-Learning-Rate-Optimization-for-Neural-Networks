"""Run sweeps side by side: one process per (method, seed), spread over the GPUs.

    python -m benchmark.pool --dataset covertype --data-dir /path/to/covertype --round sgd:1 \\
        --seeds 42 43 44 45 46 --workers 8

Every flag but ``--workers`` and ``--log-dir`` is ``python -m benchmark``'s, and
each process is that command for one method on one seed, so a run made here is
the run the notebook or the command line would have made. What changes is how
long the sweep takes, and the ``s_per_epoch`` a run records: runs that share a
GPU slow each other down.

Several sweeps can share one pool (:func:`run_sweeps`), which is how
``python -m benchmark.rounds`` runs every round and ceiling of every dataset
as one job.

Runs that already have a record are skipped, as in a sequential sweep, so a
pool that was stopped, or a job the cluster preempted and requeued, starts again
from the runs that had not finished. When the ablations are calibrated, they
wait for the Efficient Polling run of the seed they are calibrated from, which
is started first and named to every process: a process holds one seed, and left
to itself each would calibrate from its own run.

Each process writes to ``logs/<dataset>/<variant>/<method>_seed<N>.log``, the
same folder the records of that sweep sit in under ``results/``.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from .__main__ import build_experiment, parse_args, print_summary
from .methods import ABLATIONS, run_key
from .sweep import REPO_DIR, Experiment

__all__ = [
    "Sweep",
    "Task",
    "calibrated_from",
    "log_dir_for",
    "main",
    "pending",
    "run_sweeps",
    "run_tasks",
    "visible_devices",
]

# A small network steps from Python, so a process leaves most of an A100 idle.
RUNS_PER_GPU = 4


@dataclass(frozen=True)
class Task:
    """One run: one method on one seed, of the sweep numbered ``sweep``."""

    method: str
    seed: int
    sweep: int = 0

    def name(self, lr: float | None = None) -> str:
        return f"{run_key(self.method, lr)}_seed{self.seed}"


@dataclass
class Sweep:
    """One experiment's runs, and the ``python -m benchmark`` flags that make each of them."""

    experiment: Experiment
    flags: Sequence[str]  # everything but --methods and --seeds of a single run
    methods: Sequence[str]
    lr: float | None = None
    overwrite: bool = False
    log_dir: Path | None = None  # default: log_dir_for(experiment)

    def __post_init__(self) -> None:
        if self.log_dir is None:
            self.log_dir = log_dir_for(self.experiment)

    @property
    def name(self) -> str:
        """Where its logs go, under ``logs/``: ``covertype/rounds/sgd_lr1``."""
        try:
            return str(Path(self.log_dir).relative_to(REPO_DIR / "logs"))
        except ValueError:
            return str(self.log_dir)


class Handle(Protocol):
    """What the pool needs of a started run: ``subprocess.Popen`` has it."""

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


def pending(
    experiment: Experiment,
    methods: Iterable[str],
    lr: float | None = None,
    overwrite: bool = False,
    sweep: int = 0,
) -> list[Task]:
    """The runs still to make, seed by seed as in a sequential sweep.

    Efficient Polling on the calibration seed goes first when the ablations are
    calibrated from it, so they wait as little as they can.
    """
    tasks = [
        Task(method, seed, sweep)
        for seed in experiment.seeds
        for method in methods
        if overwrite or not experiment.result_path(method, seed, lr).exists()
    ]
    first = Task("efficient", experiment.ablation_seed, sweep)
    if experiment.calibrate_ablations and first in tasks:
        tasks.remove(first)
        tasks.insert(0, first)
    return tasks


def calibrated_from(experiment: Experiment, task: Task) -> Task | None:
    """The run a task reads its poll rate from, if it reads one."""
    if experiment.calibrate_ablations and task.method in ABLATIONS:
        return Task("efficient", experiment.ablation_seed, task.sweep)
    return None


def visible_devices() -> list[str]:
    """The GPUs this process may use, as ``CUDA_VISIBLE_DEVICES`` names them.

    Under SLURM that is the GPUs of the job, which may be MIG slices named by
    UUID rather than by index.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [device.strip() for device in visible.split(",") if device.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def log_dir_for(experiment: Experiment) -> Path:
    """``logs/`` mirrors ``results/``: a round's logs sit where its records do."""
    results = Path(experiment.results_dir)
    try:
        return REPO_DIR / "logs" / results.relative_to(REPO_DIR / "results")
    except ValueError:
        return REPO_DIR / "logs" / experiment.spec.key


def run_tasks(
    tasks: Sequence[Task],
    start: Callable[[Task, int], Handle],
    workers: int,
    recorded: Callable[[Task], bool],
    depends_on: Callable[[Task], Task | None] = lambda task: None,
    wait: Callable[[], None] = lambda: time.sleep(5),
    describe: Callable[[Task], str] = lambda task: f"{task.method} seed {task.seed}",
) -> dict[Task, int | None]:
    """Run ``tasks`` at most ``workers`` at a time and return each one's exit code.

    ``start(task, slot)`` starts a task on a numbered slot, free until the task
    ends, which is how a caller pins each slot to a GPU. A task that depends on
    another starts once that one, if it is among ``tasks``, has ended, and then
    only if it left a record; a task that never gets one is not started, and
    its code is ``None``.
    """
    queue = list(tasks)
    running: dict[int, tuple[Task, Handle]] = {}
    codes: dict[Task, int | None] = {}
    try:
        while queue or running:
            for slot, (task, handle) in list(running.items()):
                code = handle.poll()
                if code is not None:
                    codes[task] = code
                    del running[slot]
                    outcome = "done" if code == 0 else f"FAILED with exit code {code}"
                    print(f"{outcome}: {describe(task)}", flush=True)

            active = {task for task, _ in running.values()}
            for task in list(queue):
                dependency = depends_on(task)
                if dependency is not None and (dependency in queue or dependency in active):
                    continue
                if dependency is not None and not recorded(dependency):
                    codes[task] = None
                    queue.remove(task)
                    needed = describe(dependency)
                    print(
                        f"skipped: {describe(task)}, which needs a record of {needed}", flush=True
                    )
                    continue
                free = [slot for slot in range(workers) if slot not in running]
                if not free:
                    break
                running[free[0]] = (task, start(task, free[0]))
                queue.remove(task)
                active.add(task)

            if running:
                wait()
    finally:
        for _, handle in running.values():
            handle.terminate()
    return codes


def run_sweeps(sweeps: Sequence[Sweep], workers: int | None = None) -> dict[Task, int | None]:
    """Make every run of ``sweeps`` that has no record yet, all of them in one pool.

    Returns each run's exit code, ``None`` for an ablation whose calibration run
    never left a record. The Efficient Polling runs the ablations read go first,
    every sweep's, so no ablation waits for long.
    """
    if not sweeps:
        return {}
    cpu = str(sweeps[0].experiment.device).startswith("cpu")
    devices = [] if cpu else visible_devices()
    if workers is None:
        workers = RUNS_PER_GPU * len(devices) if devices else 1
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    queues = [
        pending(s.experiment, s.methods, s.lr, s.overwrite, sweep=index)
        for index, s in enumerate(sweeps)
    ]
    firsts = [
        first
        for index, (s, queue) in enumerate(zip(sweeps, queues, strict=True))
        if s.experiment.calibrate_ablations
        and (first := Task("efficient", s.experiment.ablation_seed, index)) in queue
    ]
    tasks = firsts + [task for queue in queues for task in queue if task not in firsts]

    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    threads = max(1, (cpus or 1) // workers)
    for s in sweeps:
        Path(s.log_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"{len(tasks)} runs to make over {len(sweeps)} sweep(s), {workers} at a time on "
        f"{', '.join(f'GPU {d}' for d in devices) or 'the CPU'}, {threads} CPU threads each",
        flush=True,
    )

    def describe(task: Task) -> str:
        prefix = f"{sweeps[task.sweep].name} | " if len(sweeps) > 1 else ""
        return f"{prefix}{task.method} seed {task.seed}"

    def start(task: Task, slot: int) -> subprocess.Popen[bytes]:
        sweep = sweeps[task.sweep]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.setdefault("OMP_NUM_THREADS", str(threads))
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[slot % len(devices)]
        # The seed the ablations calibrate from is named outright: a process
        # holds one seed, so left to itself it would read its own run.
        command = [
            sys.executable,
            "-m",
            "benchmark",
            *sweep.flags,
            "--methods",
            task.method,
            "--seeds",
            str(task.seed),
            "--calibration-seed",
            str(sweep.experiment.ablation_seed),
        ]
        where = f"GPU {env['CUDA_VISIBLE_DEVICES']}" if devices else "the CPU"
        print(f"started: {describe(task)} on {where}", flush=True)
        with open(Path(sweep.log_dir) / f"{task.name(sweep.lr)}.log", "ab") as log:
            return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    def recorded(task: Task) -> bool:
        sweep = sweeps[task.sweep]
        return sweep.experiment.result_path(task.method, task.seed, sweep.lr).exists()

    # A job the cluster stops or preempts gets SIGTERM; exiting through the
    # pool's cleanup stops the runs it started instead of orphaning them.
    previous = signal.signal(signal.SIGTERM, lambda signum, _: sys.exit(128 + signum))
    try:
        return run_tasks(
            tasks,
            start,
            workers,
            recorded=recorded,
            depends_on=lambda task: calibrated_from(sweeps[task.sweep].experiment, task),
            describe=describe,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


def unfinished(codes: dict[Task, int | None], sweeps: Sequence[Sweep]) -> str | None:
    """What to say when some runs did not finish, or ``None`` when all did."""
    failed = [task for task, code in codes.items() if code != 0]
    if not failed:
        return None
    names = ", ".join(
        f"{sweeps[t.sweep].name}/{t.name(sweeps[t.sweep].lr)}"
        if len(sweeps) > 1
        else t.name(sweeps[t.sweep].lr)
        for t in failed
    )
    logs = ", ".join(sorted({str(sweeps[t.sweep].log_dir) for t in failed}))
    return f"\n{len(failed)} of {len(codes)} runs did not finish: {names}; see {logs}"


def main(argv: list[str] | None = None) -> None:
    pool = argparse.ArgumentParser(
        prog="python -m benchmark.pool",
        description=__doc__.splitlines()[0],
        epilog="Every other flag is passed on to python -m benchmark; see its --help.",
        allow_abbrev=False,
    )
    pool.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"runs at a time (default: {RUNS_PER_GPU} per GPU, or 1 without one)",
    )
    pool.add_argument("--log-dir", default=None, help="default: logs/, mirroring results/")
    ours, forwarded = pool.parse_known_args(argv)
    if ours.workers is not None and ours.workers < 1:
        pool.error(f"--workers must be at least 1, got {ours.workers}")

    args = parse_args(forwarded)
    experiment = build_experiment(args)
    sweep = Sweep(
        experiment,
        forwarded,
        args.methods,
        args.lr,
        args.overwrite,
        Path(ours.log_dir) if ours.log_dir else None,
    )
    codes = run_sweeps([sweep], ours.workers)

    print_summary(args, experiment)
    message = unfinished(codes, [sweep])
    if message:
        sys.exit(message)


if __name__ == "__main__":
    main()
