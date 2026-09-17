# Changelog

All notable changes to the `efficient-polling-lr-scheduler` package are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The package itself is unchanged. Everything below is the benchmark around it:
four more datasets, the runs on them, and a reorganized repository.

### Added

- CIFAR-100, MNIST, Fashion-MNIST and Covertype next to CIFAR-10, read straight
  from the files their authors publish, with no torchvision and no download
  inside a run. The IDX files are accepted compressed or not, dashed or dotted,
  and a directory that shares a file's name no longer shadows it. Covertype is
  tabular: 581,012 rows of 54 cartographic features and seven cover types, on
  the published split (the first 15,120 rows to fit on, the remaining 565,892
  to test on), trained with `SimpleMLP` (512→256→128, no batch norm and no
  dropout, the same rule the CNN follows).
- The comparison on those four datasets: thirteen configurations over five
  seeds, 260 runs in `results/<dataset>/`, their figures in `images/<dataset>/`
  and the tables in `docs/results.md`. Nothing was retuned per dataset.
- `docs/methods.md`, `docs/results.md` and `docs/reproducing.md`, in English and
  in Portuguese under `docs/pt-br/`, split out of READMEs that had grown to 400
  lines each. The READMEs keep the install, the quickstart, the API and a
  five-dataset summary.
- Tests for the dataset readers, the benchmark and its figures, against files
  the tests write themselves: the suite goes from 151 to 237 and still
  downloads nothing.
- Learning-rate rounds, in `benchmark/rounds.py`, on the command line
  (`--round OPTIMIZER:RATE`) and in cells of their own in the notebook: every
  method except Adam, SPS and Armijo on one base optimizer, SGD or Adam, from
  one rate, `1`, `1e-3` or `1e-7`, recorded in `results/<dataset>/rounds/`.
  The schedules start from the round's rate, Polling and Efficient Polling
  centre their grid on it and Efficient Relative Polling starts from it, its
  `1e-1` ceiling raised to `1` in the rounds that start there.
  SPS and Armijo never read a starting rate, so they get a test of their own
  that moves their ceiling instead, on SGD only (`--ceiling RATE`,
  `results/<dataset>/ceilings/`). Nothing has been run in either yet.
- `python -m benchmark.pool`, which makes a sweep's runs side by side, one
  process per method and seed, four per GPU by default, each told which seed
  the trigger ablations are calibrated from (`--calibration-seed`, since a
  process that holds one seed would otherwise read its own run), and
  `slurm/benchmark.sbatch`, which runs one round or one ceiling of one dataset
  per cluster job and resumes from the runs left when the job is preempted or
  submitted again. Records are now written aside and moved into place, so a
  run killed while writing one leaves nothing half-written to be read back.

### Changed

- The experiment code is one package, `benchmark/`, imported by both
  `notebooks/benchmark.ipynb` and `python -m benchmark`: `datasets.py`,
  `models.py`, `methods.py`, `sweep.py` and `plots.py`. It replaces
  `examples/cifar10.py`, `examples/plot_results.py` and the notebook's own
  copies of the network, the optimizer builder, the sweep and the figures,
  which had drifted apart. Checked against the previous notebook code: runs on
  synthetic data identical field by field, identical optimizer settings for
  every method, and byte-identical CIFAR-10 figures.
- `notebooks/cifar10.ipynb` becomes `notebooks/benchmark.ipynb`, and a `DATASET`
  constant picks the dataset. It is committed without outputs (it weighed 49 MB,
  45 MB of them two embedded animations), and CI checks that with nbstripout.
- The model's input and class count, the divergence threshold, the results
  directory and the checkpoint names all follow from the dataset, so a second
  dataset cannot overwrite the first one's runs. The divergence threshold is
  `2 * ln(num_classes)` rather than a literal `2 * ln(10)`: on CIFAR-100 the old
  value sat *below* the loss a hundred-class run starts at, which would have
  rolled back every batch. CIFAR-10 keeps the number the recorded runs used.
- `SimpleCIFAR10CNN` becomes `SimpleCNN`, taking its channel and class counts
  from the dataset; on CIFAR-10 it is the same 557,898-parameter network.
- Normalization statistics are taken along the first axis of a sample, which is
  per channel for an image and per feature for a table. A 0/1 flag keeps its
  own scale (mean 0, std 1) and a feature constant over the split gets std 1
  rather than a floor of 1e-8: two of Covertype's soil types never occur in the
  15,120 training rows, and the floor sent every test row that has one set to
  1e8, with a test loss in the hundreds to show for it.
- Figures go to `images/<dataset>/` under the names the paper cites, so the
  CIFAR-10 figures moved from `images/` to `images/cifar10/`. The poll-count
  floor in `polls_per_epoch.png` follows the dataset's batches per epoch
  instead of CIFAR-10's 704.
- The trigger ablations can be calibrated to the poll rate Efficient Polling
  measured on the first seed (`--calibrate-ablations`, or
  `Experiment(calibrate_ablations=True)`), which is how the four new datasets
  were run; CIFAR-10 keeps the paper's 5%.
- `--lr` on the command line records runs as `<method>_lr<rate>`, as the
  notebook's initial-rate runs are, instead of mixing them with the table.
- The benchmark builds Polling, Efficient Polling and Efficient Relative Polling
  from the package's wrappers around `Training.optimizer` instead of its SGD
  classes, so a round can put them on Adam. On SGD the old and the new builders
  take bit-identical steps for every method, and the tables and figures drawn
  from the recorded runs come out identical. A new record keeps the base
  optimizer it stepped with, and tables and legends name the fixed rate and the
  schedules after the optimizer and rate the record holds.
- The `examples` extra becomes `benchmark`, and `dev` also installs matplotlib
  and nbstripout.
- The READMEs no longer link to the paper's LaTeX source or to a local copy of
  the base paper, neither of which is in the repository; they cite Tan et al.
  by DOI instead.

### Removed

- `images/learning_rate_factor_formula.png`, `images/training_comparison_LRs.png`
  and `images/training_comparison_losses.png`, which nothing referenced.

### Fixed

- The READMEs said Efficient Polling's backoff is immune to stalling at its
  smallest candidate. It is not: the rule that holds the backoff back records
  signal on the first poll whose candidates differ, which a single correct
  answer at chance is enough for, and CIFAR-100 stalled all five seeds that way.
  The docs now describe the failure and its mechanism.

### Note

No published CIFAR-10 number changed.

## [2.0.0] - 2026-09-07

### Changed

- **Breaking:** Relative Polling is now **Efficient Relative Polling**, so the
  method sits inside the family the package is named after. The three classes
  `RelativePollingSGD`, `RelativePollingOptimizer` and `RelativeEpochPolling`
  become `EfficientRelativePollingSGD`, `EfficientRelativePollingOptimizer` and
  `EfficientRelativeEpochPolling`; the module `relative.py` becomes
  `efficient_relative.py`. Nothing else about the method changed, and no
  measured number moved. Code written against 1.1.0 has to rename the imports;
  1.1.0 was yanked from PyPI, since it was published the day before and the old
  names never had users.
- The experiment keys `relative` and `relative_epoch` become
  `efficient_relative` and `efficient_relative_epoch`, in the notebook, in
  `examples/cifar10.py`, in the recorded runs under `results/cifar10/` and in
  the figures.
- In the figures, the short legend label for Efficient Polling was `ours`, which
  read as if the relative variant were somebody else's work. Both are ours, so
  the labels now name the methods: `Efficient`, `Eff. relative`,
  `Eff. rel. epoch`.

## [1.1.0] - 2026-09-06

Version 1.0.1 was prepared in the source tree but never published; its
README rewrite is folded in here.

### Added

- `RelativePollingSGD` / `RelativePollingOptimizer` — Relative Polling
  (experimental): three candidates `{X/m, X, X·m}` around the rate in use, the
  winner becoming the new centre, the window widening by a multiplier after
  every blind poll; an uncapped TCP-style backoff whose ceiling
  is set by blow-ups; restarts that go back to the best point seen (weights,
  optimizer state, rate and loss trend); an Adam-style loss trend (`Trend`)
  with a deviation-based spike test; and a downward tie-break on the poll
  after a restart.
- `RelativeEpochPolling` — the same method at epoch granularity, driven by
  `fit(..., epoch_polling=...)` over a plain optimizer: a poll epoch trains
  once per candidate and keeps the trial with the lowest mean training loss;
  validation accuracy judges the best point.
- `PollingOptimizer.poll()` accepts a per-call `candidates` sequence, taken in
  tie-break order, reports whether the winner was `decisive`, and scores a
  trial whose loss is not finite as a loss.
- `fit()` accepts `epoch_polling`.
- Notebook and `examples/cifar10.py`: `relative` and `relative_epoch`
  configurations, a `granularities` flag, and an initial-learning-rate
  robustness sweep with its figure.
- `results/cifar10/`: the ten Relative Polling runs (two configurations, five
  seeds) and the four initial-rate runs; figures redrawn with a fourth panel
  and `images/initial_lr_robustness.png` added. Results are in the README.

### Changed

- README.md and README(pt-br).md rewritten to match the paper: the five-seed,
  thirteen-configuration study (replacing the single-seed, three-method numbers
  from 0.1.0), the full comparison table, the cost model, the divergence-guard
  statistics and the trigger-ablation results.

### Removed

- The presentation video and the presentation slides, and every link to them in
  both READMEs. They were outdated class material that was never meant to be
  distributed with the package.

## [1.0.0] - 2026-08-02

### Added

- `SPSSGD` / `SPSOptimizer` — comparison baseline implementing the stochastic
  Polyak step-size.
- `ArmijoSGD` / `ArmijoOptimizer` — comparison baseline implementing stochastic
  Armijo backtracking line search.
- `trigger` parameter on `EfficientPollingOptimizer`/`EfficientPollingSGD`,
  exposed as `TRIGGERS = ("backoff", "fixed", "random")`: `"fixed"` and
  `"random"` are budget-matched ablations of the adaptive backoff trigger,
  used to isolate the contribution of the trigger from the poll budget itself.
- `fit`/`train_epoch` accept an optional `torch.optim.lr_scheduler`, so Adam
  and the standard schedulers (cosine annealing, step decay,
  ReduceLROnPlateau) run through the same training loop as the polling
  methods.
- `examples/plot_results.py` — redraws the paper's figures directly from the
  JSON files `examples/cifar10.py` records, so a plot can never disagree with
  the reported table.

### Changed

- `examples/cifar10.py` now runs all eleven configurations (the two polling
  methods, six comparison methods and two trigger ablations) over multiple
  seeds via `--seeds`, instead of a single-seed, three-method run.
- `notebooks/cifar10.ipynb` reworked around one shared training loop for all
  eleven configurations over five seeds (42–46), replacing the original
  single-seed, three-method notebook.

## [0.1.0] - 2026-07-30

First release: the research code from `notebooks/cifar10.ipynb` extracted into an
installable, tested library.

> **Naming history.** This library was briefly published as `efficient-polling`
> (also 0.1.0, import `efficient_polling`) before being renamed to
> `efficient-polling-lr-scheduler` and republished. The old distribution was
> removed from PyPI; nothing else changed, and the algorithms are identical.
> Note that the classes are not `torch.optim.lr_scheduler.LRScheduler` subclasses
> despite the name — they wrap the optimizer and run through `optimizer.step(closure)`.

### Added

- `PollingOptimizer` / `PollingSGD` — the replicated base polling method of Tan
  et al.: every candidate learning rate is applied as a trial step from an
  identical snapshot, and the highest-scoring one (ties favouring the smallest)
  is kept.
- `EfficientPollingOptimizer` / `EfficientPollingSGD` — the proposed extension:
  the same selection, polled on demand via exponential backoff, with the
  two-tier divergence guard (spike-triggered polls and rollback checkpoints).
- `StateSnapshot` — exact save/restore of parameters, module buffers and
  optimizer state, so every candidate departs from an identical pre-step
  condition. Gradients are preserved, since all candidates reuse the single
  gradient computed at the polled point.
- `make_closure`, `accuracy`, `negative_loss` — batch closures and selection
  criteria; polling by loss instead of accuracy is supported.
- `StepInfo`, `PollResult`, `EpochStats`, `History` — per-step and per-epoch
  telemetry, including poll counts, rollbacks, spikes and the optimizer-step
  count behind the paper's cost model.
- `train_epoch`, `evaluate`, `fit` — optional training helpers that accept either
  a polling optimizer or a plain `torch.optim.Optimizer`, so the baseline and
  both polling methods run through one loop.
- `examples/cifar10.py` — reproduces the paper's three runs from the command line.

### Notes

- Polling wraps any `torch.optim.Optimizer`, so momentum, weight decay and Adam
  all work, although the paper's results use vanilla SGD.
- Candidate learning rates are absolute and applied to every parameter group,
  which overrides per-group learning rates.
- `max_poll_interval` counts blind steps *between* polls, so the steady state is
  one poll every `max_poll_interval + 1` batches; `0` polls every batch and
  recovers the base method exactly.
