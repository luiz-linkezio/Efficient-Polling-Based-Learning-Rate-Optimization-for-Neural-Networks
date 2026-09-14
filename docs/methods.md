# How it works

[🇧🇷 Português](pt-br/methods.md) · [README](../README.md) · [Results](results.md) · [Reproducing the experiments](reproducing.md)

- [Polling](#polling-replicated-base-method), the replicated base method
- [Efficient Polling](#efficient-polling), which polls on demand
- [Efficient Relative Polling](#efficient-relative-polling), which drops the fixed grid
- [Cost model](#cost-model)

## Polling (replicated base method)

The Polling Method is from Tan et al., [Expediting Convergence via Polling Optimisation for Gradient Descent in Neural Networks](https://doi.org/10.3390/jeta4010001) (2025). At each batch, after computing the gradient `g` from weights `θ`, every candidate learning rate is applied as a trial step and the winner is kept:

```
θ̂ₖ = θ − lrₖ · g               for each lrₖ ∈ C
k*  = argmax acc(θ̂ₖ, batch)     (ties favour the smallest lr)
θ   ← θ̂ₖ*
```

The candidate set is `C = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`, spanning four orders of magnitude around the base LR. The trial updates are realized by snapshotting the model + optimizer state once, then reloading it before each candidate step, so every candidate departs from an identical pre-step condition. Because the trials overwrite the weights, the winning step is reapplied once from the snapshot, so a poll costs `N + 1 = 6` optimizer steps rather than `N = 5`.

## Efficient Polling

The selection mechanism is untouched, but a batch is polled only when needed:

1. **Adaptive polling schedule.** Let `K` be the poll interval. After a poll, if the selection is unchanged, `K ← min(2K, K_max)` (geometric backoff, capped at `K_max = 64`); if it changed, `K ← 1` (poll every batch until it stabilizes again). Between polls, a single blind SGD step uses the last selected LR. Backoff only engages once a poll has shown *signal*, that is, once the candidates' accuracies have differed. At initialization every candidate ties, and treating a tie as a stable selection would stall training at the smallest candidate forever.

   The rule has a limit. Signal is recorded once and for all, and one correct answer is enough to produce it. With many classes, a batch at chance accuracy still tells the candidates apart by a single answer now and then; after that, every tie returns the smallest candidate again, and the repetition doubles the interval. On CIFAR-100 this stalled all five seeds at `1e-5`, two of them for the whole run (see [Results](results.md#what-the-four-datasets-add)).

2. **Two-tier divergence guard.** Blind steps have no per-step validation, so a high-LR step can diverge. The guard reuses quantities already computed:
   - **Tier 2, spike-triggered polls (prevention):** if the batch loss exceeds `γ · EMA(loss)` (`γ = 3`, `β = 0.9`), poll immediately so the accuracy criterion can reject an explosive step.
   - **Tier 1, rollback checkpoints (recovery):** each poll snapshot doubles as a known-good checkpoint; if the loss is non-finite or exceeds `ℓ_rb`, twice the loss of a uniform guess, restore the checkpoint and resume polling.

   Across the five CIFAR-10 runs, spike-triggered polls numbered 378 per run on average (6.6% of all polls) and the recovery tier fired only twice in the entire study, both in the same seed. Its necessity is real, though: an early unguarded run diverged to `NaN` at epoch 33 from a single blind step at `lr = 1e-1` and never recovered, the same failure that killed most of the hand-designed schedulers in the comparison.

3. **Alternative triggers (ablation).** `trigger="fixed"` polls every `K + 1` batches at a constant interval instead of backing off, and `trigger="random"` polls each batch independently with probability `p`. Both keep rule 2 (the guard) and the selection mechanism unchanged, and both are calibrated to the poll rate the adaptive backoff measures, so a comparison against them isolates the trigger rather than the budget. See [Trigger ablation](results.md#trigger-ablation).

| Symbol | Value | Role |
|---|---|---|
| `C` | `{1e-5, …, 1e-1}` | candidate learning rates |
| `lr_init` | `1e-3` | LR before the first poll |
| `K_max` | `64` | max poll interval (backoff cap) |
| `γ` | `3` | spike threshold (tier 2) |
| `β` | `0.9` | loss-EMA decay |
| `ℓ_rb` | `2·ln(classes)`, `4.61` on CIFAR-10 | rollback threshold (tier 1) |

## Efficient Relative Polling

Both methods above choose from a *fixed* grid, and in the recorded CIFAR-10 runs the selected rate spends most of training pinned to that grid's edges: `1e-1` through the first phase, `1e-5` after the anneal. Efficient Relative Polling drops the grid. The user picks one learning rate and one multiplier `m`, and each poll tries three candidates around the rate in use; the winner becomes the new centre:

```
C_t = {X/m, X, X·m}        X ← argmax acc(θ̂, batch)
```

When a poll comes back blind (every candidate scoring the same, which batch accuracy does whenever one step moves no prediction), the next poll looks one multiplier farther out in both directions, and keeps widening until it sees a difference; a poll with signal narrows the window back. Three rules complete it:

1. **Uncapped backoff, bounded by failure.** The poll interval `k` doubles when a scheduled poll with signal keeps the rate, a blind poll leaves it unchanged, and it has no `K_max`. Every poll checkpoints the *best* point seen so far (weights, optimizer state and rate, judged by a slow trend of the loss). When a blind stretch blows up, training goes back in time to that point, `k` drops to zero and the ceiling of the next slow start becomes half the interval that blew up; growth is exponential up to the ceiling and linear above it, as in TCP congestion control.
2. **Ties keep the rate, except after a restart.** A tie carries no information, so an ordinary poll keeps `X`. The poll right after a restart breaks ties one notch *down*, because a restart has a single cause: a rate too high for blind steps.
3. **Adam-style trend.** The loss trend is a bias-corrected exponential mean with a second moment of its deviations, so a spike is a loss more than `z` deviations above the trend rather than a fixed ratio to it.

```python
from efficient_polling_lr_scheduler import EfficientRelativePollingSGD

optimizer = EfficientRelativePollingSGD(model, lr=1e-3, multiplier=10.0, lr_max=1e-1)
```

| Symbol | Default | Role |
|---|---|---|
| `lr` | `1e-3` | the one rate the user chooses |
| `m` | `10` | candidate spacing |
| `lr_min`, `lr_max` | none | optional bounds; the experiments cap at `1e-1`, the ceiling every other method is held to |
| `z` | `3` | spike threshold, in deviations above the trend |
| `ℓ_rb` | `2·ln(classes)` | blow-up threshold, as above |

`EfficientRelativeEpochPolling` applies the same window per epoch instead of per batch, driven by `fit(..., epoch_polling=...)`: it trains a whole epoch per candidate from one snapshot and keeps the one with the lowest mean training loss. It is kept as a documented negative result; see [Results](results.md#efficient-relative-polling).

## Cost model

A poll costs `N + 1` optimizer steps: one trial per candidate, plus reapplying the winner. That is 6 steps with the fixed grid of five candidates and 4 with the relative window of three. With `P` polls out of `B` batches:

```
S_eff = P·(N+1) + (B − P) = B + N·P
```

On CIFAR-10, `B = 105,600` and Efficient Polling averages `P = 5,732.2` polls, which gives `134,261` steps: a 79% reduction versus base Polling's `633,600`, exactly matching the measured table. Only 26% of those steps come from polls; the remaining 74% are ordinary SGD steps, which is why the per-epoch wall-clock time sits at the level of plain SGD. The step counts of every method on every dataset are in [Results](results.md).
