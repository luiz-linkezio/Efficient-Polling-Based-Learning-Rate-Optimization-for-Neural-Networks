"""Run a sweep's runs side by side: one process per (method, seed), spread over the GPUs.

    python -m benchmark.pool --dataset covertype --data-dir /path/to/covertype --round sgd:1 \\
        --seeds 42 43 44 45 46 --workers 8

Every flag but ``--workers`` and ``--log-dir`` is ``python -m benchmark``'s, and
each process is that command for one method on one seed, so a run made here is
the run the notebook or the command line would have made. What changes is how
long the sweep takes, and the ``s_per_epoch`` a run records: runs that share a
GPU slow each other down.

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
    "Task",
    "calibrated_from",
    "log_dir_for",
    "main",
    "pending",
    "run_tasks",
    "visible_devices",
]

# A small network steps from Python, so a process leaves most of an A100 idle.
RUNS_PER_GPU = 4


@dataclass(frozen=True)
class Task:
    """One run: one method on one seed."""

    method: str
    seed: int

    def name(self, lr: float | None = None) -> str:
        return f"{run_key(self.method, lr)}_seed{self.seed}"


class Handle(Protocol):
    """What the pool needs of a started run: ``subprocess.Popen`` has it."""

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


def pending(
    experiment: Experiment,
    methods: Iterable[str],
    lr: float | None = None,
    overwrite: bool = False,
) -> list[Task]:
    """The runs still to make, seed by seed as in a sequential sweep.

    Efficient Polling on the calibration seed goes first when the ablations are
    calibrated from it, so they wait as little as they can.
    """
    tasks = [
        Task(method, seed)
        for seed in experiment.seeds
        for method in methods
        if overwrite or not experiment.result_path(method, seed, lr).exists()
    ]
    first = Task("efficient", experiment.ablation_seed)
    if experiment.calibrate_ablations and first in tasks:
        tasks.remove(first)
        tasks.insert(0, first)
    return tasks


def calibrated_from(experiment: Experiment, task: Task) -> Task | None:
    """The run a task reads its poll rate from, if it reads one."""
    if experiment.calibrate_ablations and task.method in ABLATIONS:
        return Task("efficient", experiment.ablation_seed)
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
                    print(f"{outcome}: {task.method} seed {task.seed}", flush=True)

            active = {task for task, _ in running.values()}
            for task in list(queue):
                dependency = depends_on(task)
                if dependency is not None and (dependency in queue or dependency in active):
                    continue
                if dependency is not None and not recorded(dependency):
                    codes[task] = None
                    queue.remove(task)
                    print(
                        f"skipped: {task.method} seed {task.seed}, "
                        f"which needs a record of {dependency.method} seed {dependency.seed}",
                        flush=True,
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

    args = parse_args(forwarded)
    experiment = build_experiment(args)
    devices = [] if str(experiment.device).startswith("cpu") else visible_devices()
    workers = ours.workers
    if workers is None:
        workers = RUNS_PER_GPU * len(devices) if devices else 1
    if workers < 1:
        pool.error(f"--workers must be at least 1, got {workers}")
    log_dir = Path(ours.log_dir) if ours.log_dir else log_dir_for(experiment)
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = pending(experiment, args.methods, args.lr, args.overwrite)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    threads = max(1, (cpus or 1) // workers)
    print(
        f"{len(tasks)} runs to make, {workers} at a time on "
        f"{', '.join(f'GPU {d}' for d in devices) or 'the CPU'}, {threads} CPU threads each; "
        f"logs in {log_dir}",
        flush=True,
    )

    def start(task: Task, slot: int) -> subprocess.Popen[bytes]:
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
            *forwarded,
            "--methods",
            task.method,
            "--seeds",
            str(task.seed),
            "--calibration-seed",
            str(experiment.ablation_seed),
        ]
        where = f"GPU {env['CUDA_VISIBLE_DEVICES']}" if devices else "the CPU"
        print(f"started: {task.method} seed {task.seed} on {where}", flush=True)
        with open(log_dir / f"{task.name(args.lr)}.log", "ab") as log:
            return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    # A job the cluster stops or preempts gets SIGTERM; exiting through the
    # pool's cleanup stops the runs it started instead of orphaning them.
    previous = signal.signal(signal.SIGTERM, lambda signum, _: sys.exit(128 + signum))
    try:
        codes = run_tasks(
            tasks,
            start,
            workers,
            recorded=lambda task: experiment.result_path(task.method, task.seed, args.lr).exists(),
            depends_on=lambda task: calibrated_from(experiment, task),
        )
    finally:
        signal.signal(signal.SIGTERM, previous)

    print_summary(args, experiment)
    failed = [task for task, code in codes.items() if code != 0]
    if failed:
        names = ", ".join(task.name(args.lr) for task in failed)
        sys.exit(f"\n{len(failed)} of {len(tasks)} runs did not finish: {names}; see {log_dir}")


if __name__ == "__main__":
    main()
