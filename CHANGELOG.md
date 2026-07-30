# Changelog

All notable changes to the `efficient-polling-lr-scheduler` package are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
