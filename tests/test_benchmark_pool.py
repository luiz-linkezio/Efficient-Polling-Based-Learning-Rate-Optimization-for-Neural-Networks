"""Runs side by side: which runs a pool makes, in what order, and on which GPU.

No run trains here. The processes are stand-ins that end when the test says so
and write the record a real run would, which is all the pool reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import pool
from benchmark.__main__ import parse_args
from benchmark.pool import Task, calibrated_from, log_dir_for, pending, run_tasks
from benchmark.rounds import ROUND_METHODS
from benchmark.sweep import REPO_DIR, Experiment


class Run:
    """A stand-in process that ends, with ``code``, after ``polls`` polls."""

    def __init__(self, task: Task, polls: int = 1, code: int = 0) -> None:
        self.task, self.polls, self.code = task, polls, code
        self.terminated = False

    def poll(self) -> int | None:
        self.polls -= 1
        return self.code if self.polls < 0 else None

    def terminate(self) -> None:
        self.terminated = True


class Cluster:
    """Starts stand-in runs, keeps the records of those that succeed, and notes
    what ran at the same time."""

    def __init__(self, polls: int = 1, failing: tuple[Task, ...] = ()) -> None:
        self.polls, self.failing = polls, failing
        self.records: set[Task] = set()
        self.started: list[tuple[Task, int]] = []
        self.runs: list[Run] = []
        self.most_at_once = 0

    def start(self, task: Task, slot: int) -> Run:
        run = Run(task, self.polls, 1 if task in self.failing else 0)
        self.started.append((task, slot))
        self.runs.append(run)
        return run

    def wait(self) -> None:
        running = [run for run in self.runs if run.polls >= 0]
        self.most_at_once = max(self.most_at_once, len(running))
        for run in running:
            if run.polls == 0 and run.code == 0:
                self.records.add(run.task)

    def order(self) -> list[Task]:
        return [task for task, _ in self.started]


def experiment(tmp_path: Path, calibrate: bool = True, seeds: tuple[int, ...] = (42, 43)):
    return Experiment(
        "covertype", "/nowhere", seeds=seeds, results_dir=tmp_path, calibrate_ablations=calibrate
    )


def write_record(directory: Path, method: str, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{method}_seed{seed}.json").write_text(json.dumps({"method": method}))


# --- which runs, in what order -------------------------------------------------------


def test_a_pool_makes_the_runs_without_a_record_seed_by_seed(tmp_path: Path) -> None:
    write_record(tmp_path, "baseline", 42)

    tasks = pending(experiment(tmp_path, calibrate=False), ["baseline", "cosine"])

    assert tasks == [Task("cosine", 42), Task("baseline", 43), Task("cosine", 43)]


def test_efficient_on_the_first_seed_goes_first_when_the_ablations_read_it(tmp_path: Path) -> None:
    tasks = pending(experiment(tmp_path), ROUND_METHODS)

    assert tasks[0] == Task("efficient", 42)
    assert len(tasks) == 2 * len(ROUND_METHODS)
    assert tasks.count(Task("efficient", 42)) == 1


def test_overwriting_makes_the_recorded_runs_again(tmp_path: Path) -> None:
    write_record(tmp_path, "baseline", 42)

    tasks = pending(experiment(tmp_path, seeds=(42,)), ["baseline"], overwrite=True)

    assert tasks == [Task("baseline", 42)]


def test_only_calibrated_ablations_depend_on_a_run(tmp_path: Path) -> None:
    calibrated, fixed = experiment(tmp_path), experiment(tmp_path, calibrate=False)

    assert calibrated_from(calibrated, Task("efficient_random", 43)) == Task("efficient", 42)
    assert calibrated_from(calibrated, Task("efficient", 43)) is None
    assert calibrated_from(fixed, Task("efficient_fixed", 42)) is None


# --- running them ----------------------------------------------------------------------


def run(cluster: Cluster, tasks: list[Task], workers: int, tmp_path: Path) -> dict:
    return run_tasks(
        tasks,
        cluster.start,
        workers,
        recorded=lambda task: task in cluster.records,
        depends_on=lambda task: calibrated_from(experiment(tmp_path), task),
        wait=cluster.wait,
    )


def test_no_more_runs_than_workers_go_at_once_and_every_run_is_made(tmp_path: Path) -> None:
    tasks = [Task(method, 42) for method in ("baseline", "cosine", "step", "plateau", "polling")]
    cluster = Cluster(polls=2)

    codes = run(cluster, tasks, workers=2, tmp_path=tmp_path)

    assert codes == dict.fromkeys(tasks, 0)
    assert cluster.most_at_once == 2
    assert {slot for _, slot in cluster.started} == {0, 1}


def test_the_ablations_wait_for_the_run_they_are_calibrated_from(tmp_path: Path) -> None:
    tasks = pending(experiment(tmp_path), ["efficient_fixed", "efficient", "efficient_random"])
    cluster = Cluster(polls=3)

    codes = run(cluster, tasks, workers=4, tmp_path=tmp_path)

    order = cluster.order()
    assert order[0] == Task("efficient", 42)
    assert order.index(Task("efficient_fixed", 42)) > order.index(Task("efficient", 43))
    assert all(code == 0 for code in codes.values())


def test_an_ablation_waits_for_a_rerun_even_with_an_old_record_in_place(tmp_path: Path) -> None:
    tasks = [Task("efficient", 42), Task("efficient_fixed", 42)]
    cluster = Cluster(polls=2)
    cluster.records.add(Task("efficient", 42))

    run(cluster, tasks, workers=2, tmp_path=tmp_path)

    assert cluster.order() == tasks


def test_a_failed_run_skips_what_depends_on_it_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    efficient = Task("efficient", 42)
    tasks = [efficient, Task("efficient_fixed", 42), Task("baseline", 42)]
    cluster = Cluster(failing=(efficient,))

    codes = run(cluster, tasks, workers=1, tmp_path=tmp_path)

    assert codes == {efficient: 1, Task("efficient_fixed", 42): None, Task("baseline", 42): 0}
    output = capsys.readouterr().out
    assert "FAILED with exit code 1: efficient seed 42" in output
    assert "skipped: efficient_fixed seed 42, which needs a record of efficient seed 42" in output


def test_a_stopped_pool_stops_the_runs_it_started(tmp_path: Path) -> None:
    cluster = Cluster(polls=10)

    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_tasks(
            [Task("baseline", 42), Task("cosine", 42)],
            cluster.start,
            2,
            recorded=lambda task: False,
            wait=interrupted,
        )

    assert [r.terminated for r in cluster.runs] == [True, True]


# --- where they run and log ----------------------------------------------------------


@pytest.mark.parametrize(
    ("visible", "devices"),
    [
        ("0,1", ["0", "1"]),
        ("MIG-d2e32c43, MIG-5f1c0a9e", ["MIG-d2e32c43", "MIG-5f1c0a9e"]),
        ("", []),
    ],
)
def test_the_gpus_are_the_ones_the_job_was_given(
    visible: str, devices: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    assert pool.visible_devices() == devices


def test_the_logs_of_a_round_mirror_its_records() -> None:
    folder = REPO_DIR / "results" / "mnist" / "rounds" / "adam_lr1"
    other = Experiment("mnist", "/nowhere", results_dir=folder)

    assert log_dir_for(other) == REPO_DIR / "logs" / "mnist" / "rounds" / "adam_lr1"


def test_the_command_line_starts_one_benchmark_run_per_method_and_seed_on_alternating_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The data directory does not exist: a run that touched it here would raise."""
    started: list[tuple[list[str], dict[str, str]]] = []

    class Process(Run):
        def __init__(self, command, env, stdout, stderr) -> None:
            super().__init__(Task(command[command.index("--methods") + 1], 0), polls=0)
            started.append((command, env))
            seed = int(command[command.index("--seeds") + 1])
            write_record(tmp_path / "rounds" / "sgd_lr1", self.task.method, seed)

    monkeypatch.setattr(pool.subprocess, "Popen", Process)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(pool.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pool, "print_summary", lambda args, experiment: None)

    pool.main(
        [
            "--workers",
            "2",
            "--log-dir",
            str(tmp_path / "logs"),
            "--dataset",
            "covertype",
            "--data-dir",
            "/nowhere",
            "--device",
            "cuda",
            "--results-dir",
            str(tmp_path),
            "--round",
            "sgd:1",
            "--methods",
            "baseline",
            "efficient",
            "--seeds",
            "42",
        ]
    )

    commands = [command for command, _ in started]
    assert all(c[1:3] == ["-m", "benchmark"] for c in commands)
    runs = [parse_args(c[3:]) for c in commands]  # what the benchmark itself reads
    assert [(r.methods, r.seeds) for r in runs] == [(["efficient"], [42]), (["baseline"], [42])]
    assert all(r.round.key == "sgd_lr1" and r.results_dir == str(tmp_path) for r in runs)
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in started] == ["0", "1"]
    assert all(int(env["OMP_NUM_THREADS"]) >= 1 for _, env in started)
    assert (tmp_path / "logs" / "baseline_seed42.log").exists()
    assert "2 runs to make, 2 at a time on GPU 0, GPU 1" in capsys.readouterr().out


def test_every_run_calibrates_from_the_sweeps_first_seed_not_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process holds one seed, so the seed to calibrate from is named outright:
    left to itself, the ablation of seed 43 read a run of seed 43 that a sweep
    never calibrates from, and that was still being made."""
    started: list[list[str]] = []
    write_record(tmp_path / "rounds" / "sgd_lr1", "efficient", 42)

    class Process(Run):
        def __init__(self, command, env, stdout, stderr) -> None:
            super().__init__(Task("efficient_fixed", 0), polls=0)
            started.append(command)

    monkeypatch.setattr(pool.subprocess, "Popen", Process)
    monkeypatch.setattr(pool.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pool, "print_summary", lambda args, experiment: None)

    pool.main(
        [
            "--log-dir",
            str(tmp_path / "logs"),
            "--dataset",
            "covertype",
            "--data-dir",
            "/nowhere",
            "--device",
            "cpu",
            "--results-dir",
            str(tmp_path),
            "--round",
            "sgd:1",
            "--methods",
            "efficient_fixed",
            "--seeds",
            "42",
            "43",
        ]
    )

    runs = [parse_args(command[3:]) for command in started]
    assert [(r.seeds, r.calibration_seed) for r in runs] == [([42], 42), ([43], 42)]


def test_the_command_line_says_which_runs_did_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failing(Run):
        def __init__(self, command, env, stdout, stderr) -> None:
            super().__init__(Task("baseline", 42), polls=0, code=1)

    monkeypatch.setattr(pool.subprocess, "Popen", Failing)
    monkeypatch.setattr(pool.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pool, "print_summary", lambda args, experiment: None)
    flags = ["--data-dir", "/nowhere", "--device", "cpu", "--results-dir", str(tmp_path)]

    with pytest.raises(SystemExit, match="1 of 1 runs did not finish: baseline_seed42"):
        pool.main(["--log-dir", str(tmp_path / "logs"), *flags, "--methods", "baseline"])
