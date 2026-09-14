# Changelog

All notable changes to the `efficient-polling-lr-scheduler` package are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `examples/benchmark_datasets.py` — CIFAR-10, CIFAR-100, MNIST,
  Fashion-MNIST and Covertype, read straight from the files their authors
  publish. No torchvision and no download inside a run: the IDX files are
  accepted compressed or not, dashed or dotted, and a directory that shares a
  file's name no longer shadows it. Adding a dataset is one entry in `DATASETS`.
- Covertype is the one that is not images: 581,012 rows of 54 cartographic
  features and seven cover types, on the published split (the first 15,120 rows
  to fit on, the remaining 565,892 to test on). With no spatial axes to
  convolve over it trains `SimpleMLP` — 512→256→128, no batch norm and no
  dropout, the same rule the CNN follows — so the comparison stops being about
  convolutions. A dataset now declares an `input_shape` rather than a channel
  count and a side length, and the model follows from it.
- `--dataset` on the sweep script and on `examples/plot_results.py`, and a
  `DATASET` constant in the notebook. The model's input channels and class
  count, the divergence threshold, the results directory and the checkpoint
  names all follow from it, so a second dataset cannot overwrite the first
  one's runs. Results default to `results/<dataset>/`.
- Tests for the loaders and for what the sweep builds per dataset: 49 of them
  (the suite goes from 151 to 200),
  against files the tests write themselves, so the suite still downloads
  nothing. `numpy` joins the `dev` extra, which is what they parse with.

### Changed

- The divergence threshold is now `2 * ln(num_classes)` rather than a literal
  `2 * ln(10)`. On CIFAR-100 the old value sat *below* the loss a hundred-class
  run starts at, which would have rolled back every batch without erroring
  once. CIFAR-10 keeps the number the recorded runs used.
- `examples/cifar10.py` becomes `examples/benchmark.py` and
  `notebooks/cifar10.ipynb` becomes `notebooks/benchmark.ipynb`: neither is
  about one dataset any more. `SimpleCIFAR10CNN` becomes `SimpleCNN`, taking
  its channel and class counts from the dataset; on CIFAR-10 it is the same
  557,898-parameter network, unchanged.
- The notebook now imports its loaders from `examples/benchmark_datasets.py`
  instead of carrying its own copy, so the notebook and the command line cannot
  disagree about what they are training on.
- Normalization statistics are taken along the first axis of a sample instead of
  a hard-coded channel axis, which is per channel for an image and per feature
  for a table, from the same code. A 0/1 flag keeps its own scale (mean 0,
  std 1) and a feature constant over the split gets std 1 rather than a floor
  of 1e-8: two of Covertype's soil types never occur in the 15,120 training
  rows, and the floor sent every test row that has one set to 1e8, with a
  test loss in the hundreds to show for it. An image channel meets neither case.
- CIFAR-10 figures keep the file names the paper cites; every other dataset
  appends its own (`polls_per_epoch_mnist.png`), so a later sweep cannot
  overwrite the paper's figures.

### Note

No published result changed. Every number in the README and in the paper is
CIFAR-10, and the other four datasets are wired end to end but not yet run.

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
