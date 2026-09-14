# Results

[🇧🇷 Português](pt-br/results.md) · [README](../README.md) · [How it works](methods.md) · [Reproducing the experiments](reproducing.md)

Every configuration runs on five seeds (42–46) for 150 epochs with batch size 64, and every number is the mean ± sample standard deviation over the five runs. Test metrics come from each run's own best validation epoch, so the test set never selects anything. The hyperparameters are the same on every dataset: nothing was retuned. [Reproducing the experiments](reproducing.md) has the full setup.

- [All five datasets](#all-five-datasets)
- [CIFAR-10](#cifar-10), the paper's benchmark: the full table, the learned schedule, the trigger ablation and Efficient Relative Polling
- [CIFAR-100, Fashion-MNIST, MNIST and Covertype](#cifar-100-fashion-mnist-mnist-and-covertype): what the other four datasets add, with their tables and figures

## All five datasets

Test accuracy. The best method on each dataset is in bold.

| Method | CIFAR-10 | CIFAR-100 | Fashion-MNIST | MNIST | Covertype |
|---|---|---|---|---|---|
| SGD (fixed `1e-3`) | 56.06% ± 1.76% | 20.21% ± 1.15% | 85.43% ± 0.33% | 98.46% ± 0.06% | 56.38% ± 0.45% |
| Adam (`1e-3`) | 81.83% ± 0.55% | 48.98% ± 0.63% | **92.70% ± 0.31%** | **99.45% ± 0.10%** | 75.03% ± 0.49% |
| SGD + cosine annealing | 81.73% ± 1.10% | 51.50% ± 1.83% | 92.41% ± 0.07% | 98.09% ± 2.79% | 75.64% ± 0.73% |
| SGD + step decay | 80.35% ± 3.37% | 49.77% ± 2.71% | 92.16% ± 0.32% | 99.30% ± 0.07% | 75.58% ± 0.29% |
| SGD + ReduceLROnPlateau | 83.02% ± 1.12% | 53.03% ± 0.51% | 92.02% ± 0.11% | 99.31% ± 0.06% | 74.89% ± 0.87% |
| SPS (Polyak) | 82.86% ± 0.35% | 52.98% ± 0.56% | 91.88% ± 0.19% | 99.11% ± 0.05% | 75.62% ± 0.59% |
| Armijo line search | 84.09% ± 0.38% | 52.57% ± 0.62% | 91.91% ± 0.37% | 99.39% ± 0.03% | 74.68% ± 0.28% |
| Polling (base paper) | 83.83% ± 0.14% | 52.99% ± 0.55% | 91.83% ± 0.36% | 98.14% ± 0.07% | 75.15% ± 0.71% |
| Efficient Polling (ours) | 83.76% ± 0.72% | 25.07% ± 23.31% | 91.84% ± 0.25% | 98.82% ± 0.48% | **75.79% ± 0.95%** |
| ↳ ablation: fixed interval | 68.91% ± 32.91% | 32.11% ± 28.41% | 75.35% ± 36.51% | 81.37% ± 39.14% | 75.24% ± 0.58% |
| ↳ ablation: random trigger | 84.02% ± 0.54% | 42.53% ± 23.22% | 91.52% ± 0.46% | 63.64% ± 48.54% | 74.99% ± 0.60% |
| Efficient Relative Polling (ours, per batch) | **84.11% ± 0.41%** | **53.05% ± 0.57%** | 91.70% ± 0.24% | 98.59% ± 0.09% | 75.41% ± 0.60% |
| ↳ per epoch | 47.99% ± 1.91% | 21.11% ± 0.61% | 80.98% ± 3.43% | 98.12% ± 0.72% | 60.47% ± 0.56% |

Optimizer steps of the polling methods, as a multiple of plain SGD's. Every other method takes one step per batch, except Armijo, whose backtracking adds up to 2.6× on MNIST.

| Method | CIFAR-10 | CIFAR-100 | Fashion-MNIST | MNIST | Covertype |
|---|---|---|---|---|---|
| Polling (base paper) | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |
| Efficient Polling (ours) | 1.27 | 1.18 | 1.42 | 2.23 | 1.43 |
| Efficient Relative Polling (ours, per batch) | 1.21 | 1.08 | 1.06 | 1.12 | 1.04 |

What the five datasets show:

1. **Efficient Relative Polling matches base Polling on every dataset, at a sixth of its cost or less.** It is above base Polling on four datasets and within one standard deviation of it on Fashion-MNIST, and it is the best method on CIFAR-10 and CIFAR-100.
2. **No method is best everywhere.** Adam leads on MNIST and Fashion-MNIST, where the three polling methods trail it by 0.6 to 1.3 points; on MNIST base Polling even ends below the fixed-rate baseline. Efficient Polling is the best method on Covertype.
3. **Efficient Polling can stall at its smallest candidate.** On CIFAR-100 it did on all five seeds, and two never trained. Base Polling and Efficient Relative Polling never stalled, on any seed of any dataset. [The mechanism is below](#what-the-four-datasets-add).
4. **The schedulers that start at `1e-1` diverge on four of the five datasets**, and the table credits them with their best checkpoint before the divergence.
5. **Per epoch stays a negative result.** Efficient Relative Polling per epoch is last or second to last on four of the five datasets.

## CIFAR-10

The table below reproduces the paper's Table I, eleven configurations, plus the two Efficient Relative Polling configurations added after the paper: 704 batches per epoch, 105,600 per run.

| Method | Best Val | Test Acc | Test Loss | Polled | Optimizer Steps | s/Epoch |
|---|---|---|---|---|---|---|
| SGD (fixed `1e-3`) | 56.50% ± 1.81% | 56.06% ± 1.76% | 1.2274 ± 0.0455 | n/a | 105,600 | 2.86 ± 0.01 |
| Adam (`1e-3`) | 82.75% ± 0.42% | 81.83% ± 0.55% | 1.1310 ± 0.3572 | n/a | 105,600 | 3.03 ± 0.01 |
| SGD + cosine annealing | 82.44% ± 1.13% | 81.73% ± 1.10% | **0.6727 ± 0.0645** | n/a | 105,600 | 2.85 ± 0.08 |
| SGD + step decay | 80.61% ± 3.65% | 80.35% ± 3.37% | 0.7732 ± 0.2402 | n/a | 105,600 | 2.76 ± 0.00 |
| SGD + ReduceLROnPlateau | 83.51% ± 1.44% | 83.02% ± 1.12% | 0.8942 ± 0.1624 | n/a | 105,600 | 2.77 ± 0.00 |
| SPS (Polyak) | 83.71% ± 0.45% | 82.86% ± 0.35% | 1.0695 ± 0.2744 | n/a | 105,600 | 2.96 ± 0.00 |
| Armijo line search | **84.68% ± 0.56%** | **84.09% ± 0.38%** | 1.2587 ± 0.1395 | 100% | 106,090 | 4.11 ± 0.12 |
| Polling (base paper) | 84.08% ± 0.81% | 83.83% ± 0.14% | 0.6821 ± 0.0091 | 100% | 633,600 | 8.05 ± 0.05 |
| **Efficient Polling (ours)** | 84.37% ± 0.91% | 83.76% ± 0.72% | 0.7392 ± 0.0201 | **5.43%** | **134,261** | **2.81 ± 0.02** |
| ↳ ablation: fixed interval | 69.20% ± 33.49% | 68.91% ± 32.91% | 1.0999 ± 0.6782 | 5.43% | 134,286 | 2.81 ± 0.01 |
| ↳ ablation: random trigger | 84.45% ± 0.61% | 84.02% ± 0.54% | 0.8116 ± 0.0235 | 5.77% | 136,085 | 2.83 ± 0.01 |
| **Efficient Relative Polling (ours, per batch)** | 84.68% ± 0.77% | **84.11% ± 0.41%** | 0.9803 ± 0.1398 | 7.25% | 127,379 | 2.82 ± 0.25 |
| ↳ per epoch | 48.18% ± 2.61% | 47.99% ± 1.91% | 1.4184 ± 0.0450 | 61.33% | 234,432 | 6.47 ± 0.52 |

The three schedulers start at `1e-1` (the top of the candidate set) rather than at the baseline `1e-3`, since a decay schedule needs somewhere to decay from; SPS and Armijo are capped at that same `1e-1`, so no method may take a step the others were never allowed to consider. Despite that, cosine annealing, step decay and ReduceLROnPlateau diverge to `NaN` around epoch 28 on most of their seeds (5/5, 4/5 and 1/5 respectively). The table still credits them with their best pre-divergence checkpoint, since every method is scored at its own best validation epoch. Both Polling variants hold that same `1e-1` for roughly thirty epochs across their 25 combined runs without a single failure: what breaks the schedulers is not the rate itself but the absence of a per-step check on it.

Both polling methods autonomously discover the same **two-phase schedule** entirely from batch-level feedback: the mean selected LR converges to `≈1e-1` within the first epoch, holds there for ~30 epochs, then collapses toward `≈1e-5` for fine refinement near convergence. Base Polling completes the anneal between epochs 42–45, Efficient Polling more gradually, between epochs 49–78, while its interval has not yet reset to one on every batch.

| | |
|---|---|
| ![Loss curves](../images/cifar10/training_comparison_losses_all.png) | ![Learning-rate trajectories](../images/cifar10/training_comparison_LRs_all.png) |
| Training & validation loss, all thirteen configurations. | Mean selected LR per epoch, symlog, with a cross marking divergence. |

![Polls per epoch](../images/cifar10/polls_per_epoch.png)

Polls concentrate exactly where the schedule changes: the steady-state floor is `704 / (K_max + 1) ≈ 11` polls/epoch, the median over all epochs is 16.4, and the count peaks at 231 in epoch 32, the exact epoch the selected LR begins collapsing from `1e-1` to `1e-5`, where disagreeing polls keep resetting the interval to one. This is the mechanism that lets ~5% of the polls recover the full two-phase schedule, at the cost the [cost model](methods.md#cost-model) predicts.

### Trigger ablation

Two control variants isolate the contribution of the adaptive backoff by replacing only the trigger, keeping the candidate set, selection rule and two-tier guard unchanged: `fixed` polls at a constant interval (`K = 19`) and `random` polls each batch independently with probability `p = 0.05`, both calibrated to the ~5% rate the backoff measures.

On final accuracy the **random trigger is competitive**: 84.02% test vs. 83.76% for the backoff, well inside the seed-to-seed spread. This is an honest negative result for the strong reading of the claim: at this budget, distributing polls uniformly at random is enough to track the schedule, provided the guard absorbs the cost of arriving late. The supported claim is the weaker one. The backoff reaches the same quality while spending its polls where they carry information (16.4 polls in a median epoch vs. a peak of 231 at the transition, a 14× ratio, against a flat ~40/epoch for both controls), needing **~2.4× fewer** spike-triggered guard interventions (378 vs. 919 and 888) and slightly fewer optimizer steps.

The **fixed-interval control exposes a real failure mode**: on 4/5 seeds it matches the other variants (84.17% ± 0.96% val), but on the remaining seed it never leaves the initialization plateau, ending at ~10% (random-guess) accuracy. At initialization every candidate ties, the poll keeps returning the smallest candidate LR by the tie-break rule, and at `1e-5` the weights move too little to ever break the tie. On CIFAR-10 the adaptive backoff avoids it, because it polls every batch until the candidates first differ. That protection turned out weaker than it looks here: with a hundred classes a single correct answer is enough to tell the candidates apart, and on CIFAR-100 the backoff stalled [the same way](#what-the-four-datasets-add).

### Efficient Relative Polling

Same protocol, five seeds, candidates capped at `1e-1`:

| Method | Best Val | Test Acc | Test Loss | Polled | Optimizer Steps | s/Epoch |
|---|---|---|---|---|---|---|
| Efficient Polling (fixed grid) | 84.37% ± 0.91% | 83.76% ± 0.72% | 0.7392 ± 0.0201 | 5.43% | 134,261 | 2.81 ± 0.02 |
| **Efficient Relative Polling, per batch** | 84.68% ± 0.77% | **84.11% ± 0.41%** | 0.9803 ± 0.1398 | 7.25% ± 6.25% | 127,379 | 2.82 ± 0.25 |
| Efficient Relative Polling, per epoch | 48.18% ± 2.61% | 47.99% ± 1.91% | 1.4184 ± 0.0450 | 61.33% | 234,432 | 6.47 ± 0.52 |

Per batch, Efficient Relative Polling reaches the test accuracy of Armijo backtracking (84.09%, the best comparison method) at the cost of plain SGD, polling 7% of batches; a poll costs 4 optimizer steps instead of 6, so it takes fewer steps than Efficient Polling. It finds the same first phase at `1e-1`, then settles at `1e-2` from epoch ~40 to the end instead of annealing to `1e-5`: once batch accuracy saturates the criterion goes blind, ties keep the rate, and no blow-up forced a notch down (one restart in five runs). That plateau is what costs it test loss. Training continues at `1e-2` on a training set it already fits, so its predictions grow overconfident. The poll fraction varies by seed (1.8% to 16.3%): a seed that keeps hunting between `1e-1` and `1e-2` resets its interval on every move.

**The one rate the user picks does not have to be right.** Started two decades below or above the default, on seed 42:

| Start | Efficient Polling (grid pinned to the start) | Efficient Relative Polling |
|---|---|---|
| `1e-5` | 10.04%, never leaves `1e-7` | 84.72% |
| `1e-3` (default) | 83.97% | 84.67% |
| `1e-1` | 9.98%, explodes at `10` | 84.62% |

From either start the relative window is at `1e-1` within the first epoch and reproduces the same schedule.

![Initial learning rate robustness](../images/cifar10/initial_lr_robustness.png)

**Per epoch is a negative result.** `EfficientRelativeEpochPolling`, driven by `fit(..., epoch_polling=...)`, trains a whole epoch per candidate from one snapshot and keeps the one with the lowest mean training loss. It walks the rate down to `1e-5` within thirty epochs and stalls at 48%; selecting on end-of-epoch validation accuracy instead did the same, to `1e-8` and 45%. Any single-epoch score rewards the smoothness of a small step over the progress of a large one, and a blind epoch at `1e-1` has none of the per-step protection the batch method gets from its spike polls. The granularity that works is the batch.

## CIFAR-100, Fashion-MNIST, MNIST and Covertype

The protocol is CIFAR-10's, with three differences:

- **Ablation calibration.** Both trigger ablations poll at the rate Efficient Polling measured on seed 42 of the same dataset, where CIFAR-10's used the paper's 5%:

  | Dataset | Poll rate | Fixed interval `K` |
  |---|---|---|
  | CIFAR-100 | 10.11% | 9 |
  | Fashion-MNIST | 9.26% | 10 |
  | MNIST | 23.56% | 3 |
  | Covertype | 8.26% | 11 |

- **No wall-clock column.** These runs shared one GPU, three at a time, so their seconds per epoch are comparable neither with each other nor with CIFAR-10's. Optimizer steps are the cost measure here.
- **Covertype's split.** Validation is drawn from the first 15,120 rows, which hold 2,160 rows of each class; the test set is the remaining 565,892, where one class is about half of the rows. Every method loses 12 to 14 points from validation to test.

### What the four datasets add

**Efficient Polling stalls at its smallest candidate on CIFAR-100.** Every seed started with training accuracy below 2%, twice chance, with the selected rate at the `1e-5` floor for most of that time:

| Seed | Epochs below 2% training accuracy | Polled | Test accuracy |
|---|---|---|---|
| 42 | 35 | 10.11% | 52.90% |
| 43 | 128 | 2.13% | 41.79% |
| 44 | 136 | 2.17% | 27.68% |
| 45 | 150 | 1.98% | 1.25% |
| 46 | 150 | 2.02% | 1.73% |

The cause is the backoff's signal rule meeting a hundred classes. The rule lets the poll interval grow only after some poll has seen the candidates' batch accuracies differ, and it records that once. At chance, a batch of 64 has no correct answer 53% of the time and at most one 87% of the time, so the candidates tie on almost every poll, and a tie goes to the smallest candidate. Sooner or later one candidate scores one answer more than another and the rule records signal. From then on every tie returns `1e-5` again, which the backoff reads as a stable choice, doubling the interval up to its cap. Base Polling polls every batch, and Efficient Relative Polling keeps its rate on a tie and widens its window when blind; neither stalled.

**The trigger ablations stall more often, and on more datasets.** Runs, out of five, that started with training accuracy below twice chance for at least 20 epochs; in parentheses, those that never left it:

| Method | CIFAR-10 | CIFAR-100 | Fashion-MNIST | MNIST | Covertype |
|---|---|---|---|---|---|
| Polling (base paper) | 0 | 0 | 0 | 0 | 0 |
| Efficient Polling (ours) | 0 | 5 (2) | 0 | 1 | 0 |
| ↳ ablation: fixed interval | 1 (1) | 2 (2) | 2 (1) | 3 (1) | 0 |
| ↳ ablation: random trigger | 2 | 1 (1) | 2 | 2 (2) | 0 |
| Efficient Relative Polling (ours, per batch) | 0 | 0 | 0 | 0 | 0 |

That is what their large deviations in the table above come from. On Fashion-MNIST and MNIST the controls stall where the backoff does not, at the same poll rate: the backoff polls every batch until the candidates first differ and resets its interval whenever the choice changes, while the controls keep their schedule whatever the poll finds. Covertype, with seven classes, stalled nothing.

**The schedulers diverge on four of the five datasets.** Runs, out of five, whose validation loss became `NaN`:

| Method | CIFAR-10 | CIFAR-100 | Fashion-MNIST | MNIST | Covertype |
|---|---|---|---|---|---|
| SGD + cosine annealing | 5 | 3 | 0 | 2 | 0 |
| SGD + step decay | 4 | 4 | 1 | 3 | 0 |
| SGD + ReduceLROnPlateau | 1 | 0 | 0 | 0 | 0 |

One MNIST cosine seed diverged at epoch 3 and keeps only its 93.10% checkpoint, which is where that row's 2.79-point deviation comes from.

**Efficient Polling costs more on MNIST.** It polls 24.70% of MNIST's batches and takes 2.23× the optimizer steps of SGD. Late in training its choice keeps changing, and every change resets the interval to one; it also rolls back 12 times per run, against fewer than one on CIFAR-10. Efficient Relative Polling polls 4.17% of the same batches.

### CIFAR-100

| Method | Best Val | Test Acc | Test Loss | Polled | Optimizer Steps |
|---|---|---|---|---|---|
| SGD (fixed `1e-3`) | 20.28% ± 0.79% | 20.21% ± 1.15% | 3.3492 ± 0.0524 | n/a | 105,600 |
| Adam (`1e-3`) | 49.55% ± 1.30% | 48.98% ± 0.63% | 2.1944 ± 0.1808 | n/a | 105,600 |
| SGD + cosine annealing | 51.42% ± 1.86% | 51.50% ± 1.83% | 4.0845 ± 1.3127 | n/a | 105,600 |
| SGD + step decay | 49.14% ± 2.69% | 49.77% ± 2.71% | 3.2161 ± 1.1185 | n/a | 105,600 |
| SGD + ReduceLROnPlateau | 52.61% ± 0.40% | 53.03% ± 0.51% | 2.9802 ± 1.1601 | n/a | 105,600 |
| SPS (Polyak) | 52.87% ± 0.52% | 52.98% ± 0.56% | 2.3980 ± 0.1645 | n/a | 105,600 |
| Armijo line search | 52.11% ± 0.85% | 52.57% ± 0.62% | 5.5351 ± 0.1655 | 100% | 105,806 |
| Polling (base paper) | 52.97% ± 0.86% | 52.99% ± 0.55% | 3.5409 ± 0.0485 | 100% | 633,600 |
| Efficient Polling (ours) | 25.03% ± 23.06% | 25.07% ± 23.31% | 3.7104 ± 0.8980 | 3.68% ± 3.59% | 125,042 |
| ↳ ablation: fixed interval | 32.87% ± 29.13% | 32.11% ± 28.41% | 4.5221 ± 0.1168 | 10.58% ± 0.53% | 161,452 |
| ↳ ablation: random trigger | 42.74% ± 23.39% | 42.53% ± 23.22% | 4.4912 ± 0.0994 | 11.57% ± 0.77% | 166,695 |
| Efficient Relative Polling (ours, per batch) | 53.28% ± 0.90% | 53.05% ± 0.57% | 5.1240 ± 0.4442 | 3.32% ± 0.72% | 113,688 |
| ↳ per epoch | 20.67% ± 0.74% | 21.11% ± 0.61% | 3.2956 ± 0.0268 | 64.67% ± 8.78% | 240,768 |

| | | |
|---|---|---|
| ![CIFAR-100 losses](../images/cifar100/training_comparison_losses_all.png) | ![CIFAR-100 learning rates](../images/cifar100/training_comparison_LRs_all.png) | ![CIFAR-100 polls per epoch](../images/cifar100/polls_per_epoch.png) |

### Fashion-MNIST

| Method | Best Val | Test Acc | Test Loss | Polled | Optimizer Steps |
|---|---|---|---|---|---|
| SGD (fixed `1e-3`) | 86.17% ± 0.37% | 85.43% ± 0.33% | 0.4090 ± 0.0086 | n/a | 126,600 |
| Adam (`1e-3`) | 93.61% ± 0.36% | 92.70% ± 0.31% | 0.7485 ± 0.1454 | n/a | 126,600 |
| SGD + cosine annealing | 92.65% ± 0.42% | 92.41% ± 0.07% | 0.5876 ± 0.0809 | n/a | 126,600 |
| SGD + step decay | 92.65% ± 0.30% | 92.16% ± 0.32% | 0.4567 ± 0.1217 | n/a | 126,600 |
| SGD + ReduceLROnPlateau | 92.76% ± 0.33% | 92.02% ± 0.11% | 0.3754 ± 0.0962 | n/a | 126,600 |
| SPS (Polyak) | 92.63% ± 0.44% | 91.88% ± 0.19% | 0.3518 ± 0.0705 | n/a | 126,600 |
| Armijo line search | 92.74% ± 0.34% | 91.91% ± 0.37% | 0.4926 ± 0.2230 | 100% | 129,729 |
| Polling (base paper) | 92.56% ± 0.39% | 91.83% ± 0.36% | 0.2659 ± 0.0316 | 100% | 759,600 |
| Efficient Polling (ours) | 92.57% ± 0.35% | 91.84% ± 0.25% | 0.3177 ± 0.0595 | 8.46% ± 1.33% | 180,129 |
| ↳ ablation: fixed interval | 75.97% ± 36.74% | 75.35% ± 36.51% | 0.6879 ± 0.9050 | 9.38% ± 0.59% | 185,943 |
| ↳ ablation: random trigger | 92.16% ± 0.43% | 91.52% ± 0.46% | 0.3047 ± 0.0707 | 9.79% ± 0.86% | 188,574 |
| Efficient Relative Polling (ours, per batch) | 92.27% ± 0.22% | 91.70% ± 0.24% | 0.2574 ± 0.0061 | 1.87% ± 0.21% | 133,656 |
| ↳ per epoch | 81.71% ± 3.41% | 80.98% ± 3.43% | 0.5207 ± 0.0917 | 57.73% ± 12.85% | 272,106 |

| | | |
|---|---|---|
| ![Fashion-MNIST losses](../images/fashion_mnist/training_comparison_losses_all.png) | ![Fashion-MNIST learning rates](../images/fashion_mnist/training_comparison_LRs_all.png) | ![Fashion-MNIST polls per epoch](../images/fashion_mnist/polls_per_epoch.png) |

### MNIST

| Method | Best Val | Test Acc | Test Loss | Polled | Optimizer Steps |
|---|---|---|---|---|---|
| SGD (fixed `1e-3`) | 98.35% ± 0.20% | 98.46% ± 0.06% | 0.0460 ± 0.0029 | n/a | 126,600 |
| Adam (`1e-3`) | 99.52% ± 0.07% | 99.45% ± 0.10% | 0.0385 ± 0.0075 | n/a | 126,600 |
| SGD + cosine annealing | 97.90% ± 3.29% | 98.09% ± 2.79% | 0.0653 ± 0.0900 | n/a | 126,600 |
| SGD + step decay | 99.31% ± 0.15% | 99.30% ± 0.07% | 0.0249 ± 0.0035 | n/a | 126,600 |
| SGD + ReduceLROnPlateau | 99.30% ± 0.11% | 99.31% ± 0.06% | 0.0231 ± 0.0036 | n/a | 126,600 |
| SPS (Polyak) | 99.11% ± 0.12% | 99.11% ± 0.05% | 0.0353 ± 0.0047 | n/a | 126,600 |
| Armijo line search | 99.36% ± 0.16% | 99.39% ± 0.03% | 0.0263 ± 0.0028 | 100% | 333,784 |
| Polling (base paper) | 97.98% ± 0.18% | 98.14% ± 0.07% | 0.0578 ± 0.0051 | 100% | 759,600 |
| Efficient Polling (ours) | 98.74% ± 0.48% | 98.82% ± 0.48% | 0.0349 ± 0.0149 | 24.70% ± 2.76% | 282,911 |
| ↳ ablation: fixed interval | 81.18% ± 39.11% | 81.37% ± 39.14% | 0.4911 ± 1.0249 | 26.32% ± 1.18% | 293,174 |
| ↳ ablation: random trigger | 63.66% ± 48.35% | 63.64% ± 48.54% | 0.9414 ± 1.2544 | 26.18% ± 2.43% | 292,349 |
| Efficient Relative Polling (ours, per batch) | 98.45% ± 0.16% | 98.59% ± 0.09% | 0.0420 ± 0.0022 | 4.17% ± 2.24% | 142,379 |
| ↳ per epoch | 97.84% ± 0.81% | 98.12% ± 0.72% | 0.0576 ± 0.0220 | 60.27% ± 13.94% | 277,676 |

| | | |
|---|---|---|
| ![MNIST losses](../images/mnist/training_comparison_losses_all.png) | ![MNIST learning rates](../images/mnist/training_comparison_LRs_all.png) | ![MNIST polls per epoch](../images/mnist/polls_per_epoch.png) |

### Covertype

| Method | Best Val | Test Acc | Test Loss | Polled | Optimizer Steps |
|---|---|---|---|---|---|
| SGD (fixed `1e-3`) | 70.46% ± 0.92% | 56.38% ± 0.45% | 0.9896 ± 0.0083 | n/a | 31,950 |
| Adam (`1e-3`) | 87.84% ± 0.68% | 75.03% ± 0.49% | 1.6219 ± 0.2428 | n/a | 31,950 |
| SGD + cosine annealing | 88.21% ± 0.66% | 75.64% ± 0.73% | 0.9932 ± 0.0298 | n/a | 31,950 |
| SGD + step decay | 87.66% ± 0.61% | 75.58% ± 0.29% | 0.7140 ± 0.0341 | n/a | 31,950 |
| SGD + ReduceLROnPlateau | 87.17% ± 0.83% | 74.89% ± 0.87% | 0.6892 ± 0.0127 | n/a | 31,950 |
| SPS (Polyak) | 88.08% ± 0.58% | 75.62% ± 0.59% | 1.0082 ± 0.1284 | n/a | 31,950 |
| Armijo line search | 87.93% ± 0.69% | 74.68% ± 0.28% | 1.2712 ± 0.1001 | 100% | 32,079 |
| Polling (base paper) | 88.01% ± 0.48% | 75.15% ± 0.71% | 1.0818 ± 0.0406 | 100% | 191,700 |
| Efficient Polling (ours) | 88.00% ± 0.67% | 75.79% ± 0.95% | 1.1289 ± 0.1101 | 8.63% ± 0.69% | 45,733 |
| ↳ ablation: fixed interval | 88.08% ± 0.83% | 75.24% ± 0.58% | 1.0737 ± 0.1598 | 8.50% ± 0.04% | 45,535 |
| ↳ ablation: random trigger | 87.70% ± 0.65% | 74.99% ± 0.60% | 1.0909 ± 0.0787 | 8.45% ± 0.14% | 45,445 |
| Efficient Relative Polling (ours, per batch) | 87.92% ± 0.75% | 75.41% ± 0.60% | 1.1716 ± 0.1410 | 1.94% ± 0.26% | 33,266 |
| ↳ per epoch | 74.12% ± 1.03% | 60.47% ± 0.56% | 0.8831 ± 0.0049 | 62.40% ± 9.58% | 71,398 |

| | | |
|---|---|---|
| ![Covertype losses](../images/covertype/training_comparison_losses_all.png) | ![Covertype learning rates](../images/covertype/training_comparison_LRs_all.png) | ![Covertype polls per epoch](../images/covertype/polls_per_epoch.png) |
