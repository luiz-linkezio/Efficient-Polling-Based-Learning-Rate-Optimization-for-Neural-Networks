# Changelog

All notable changes to the `efficient-polling-lr-scheduler` package are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [1.0.1] - 2026-08-03

### Changed

- README.md and README(pt-br).md rewritten to match `docs/main.tex`: the
  five-seed, eleven-configuration study (replacing the single-seed, three-method
  numbers from 0.1.0), the full comparison table, the cost model, the
  divergence-guard statistics and the trigger-ablation results. No code changes.

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
