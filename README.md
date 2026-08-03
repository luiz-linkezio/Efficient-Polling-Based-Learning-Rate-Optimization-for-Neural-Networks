# Efficient Polling-Based Learning Rate Optimization for Neural Networks

> Get the accuracy of polling-based learning-rate selection at essentially the cost of plain SGD.

[![PyPI](https://img.shields.io/pypi/v/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![Python](https://img.shields.io/pypi/pyversions/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![License](https://img.shields.io/pypi/l/efficient-polling-lr-scheduler.svg)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/LICENSE)

```bash
pip install efficient-polling-lr-scheduler
```

[🇧🇷 Versão em português](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/README(pt-br).md) · [🎥 Presentation video](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/videos/apresentação.mp4) · [📄 Paper (LaTeX source)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/main.tex)

This repository replicates the **Polling Method** of Tan et al. on CIFAR-10 and introduces **Efficient Polling**, a novel extension that recovers the same learning-rate schedule — and the same accuracy — while polling only **5.43% of batches**, cutting optimizer steps by 79% and per-epoch wall-clock time from 8.05s to 2.81s, the cost of plain SGD. The method is benchmarked over **five random seeds** against **eight comparison methods** — Adam, three schedulers, SPS, Armijo backtracking, and the replicated base Polling method — plus two ablations that isolate the contribution of the adaptive polling trigger. All methods ship as a PyTorch package.

---

## TL;DR

The learning rate is the single most influential hyperparameter in gradient-based training. Instead of picking it by hand or by a fixed schedule, **polling** tests several candidate learning rates at every batch and keeps the one that most improves batch accuracy. It works remarkably well, but it multiplies training time by the number of candidates.

**Efficient Polling** observes that the polled choice is highly redundant — within each training phase consecutive polls pick the same learning rate — and polls *on demand* instead: an exponential-backoff schedule doubles the gap between polls while the selection is stable, and a two-tier divergence guard protects the unpolled steps.

| Method | Best Val | Test Acc | Test Loss | Polled Batches | s/Epoch |
|---|---|---|---|---|---|
| Baseline (fixed SGD, `1e-3`) | 56.50% | 56.06% | 1.2274 | — | 2.86 |
| Polling (base paper) | 84.08% | 83.83% | 0.6821 | 100% | 8.05 |
| **Efficient Polling (ours)** | **84.37%** | 83.76% | 0.7392 | **5.43%** | **2.81** |

*150 epochs, mean ± sample standard deviation over five seeds (42–46), NVIDIA RTX 5070. The full eleven-way comparison — Adam, three schedulers, SPS, Armijo backtracking and two trigger-ablation variants — is in [Results](#results).*

Efficient Polling **matches** the base method's accuracy (within 0.1 pp on test) while cutting optimizer steps by **79%** and per-epoch time from 8.05s to **2.81s** — a 2.9× speed-up over base Polling, indistinguishable from plain SGD's 2.86s.

---

## Quickstart

```bash
pip install efficient-polling-lr-scheduler
```

Polling needs to re-evaluate the model to score a candidate step, so instead of the bare `optimizer.step()` you pass a **closure** that returns `(loss, score)` — the same contract as `torch.optim.LBFGS`, plus the score to maximize. `make_closure` builds it for you:

```python
import torch
from efficient_polling_lr_scheduler import EfficientPollingSGD, make_closure

model = MyModel().to(device)
loss_fn = torch.nn.CrossEntropyLoss()

# Candidate LRs default to {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} around lr.
optimizer = EfficientPollingSGD(model, lr=1e-3)

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    info = optimizer.step(make_closure(model, loss_fn, inputs, targets))
    # info.lr, info.loss, info.polled, info.spike, info.rolled_back, ...
```

No learning-rate schedule, no warmup, no tuning: the learning rate is *measured*. Swap `EfficientPollingSGD` for `PollingSGD` to get the base method (polls every batch), or wrap any optimizer you like:

```python
from efficient_polling_lr_scheduler import EfficientPollingOptimizer

optimizer = EfficientPollingOptimizer(
    torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9),
    candidate_lrs=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    module=model,          # so BatchNorm buffers are restored between trials
    max_poll_interval=64,  # backoff cap; 0 polls every batch
)
```

The package also ships two comparison baselines used in the paper — the stochastic Polyak step-size (SPS) and Armijo backtracking line search, both of which derive the step size from the current batch at no extra forward pass:

```python
from efficient_polling_lr_scheduler import SPSSGD, ArmijoSGD

optimizer = SPSSGD(model, lr=1e-3, max_lr=0.1)
# or
optimizer = ArmijoSGD(model, lr=1e-3, lr_max=0.1)
```

The optional training helpers run a full comparison in a few lines, and accept a plain optimizer (and an optional `torch.optim.lr_scheduler`) too — so the baseline, Adam and the standard schedulers go through the same loop:

```python
from efficient_polling_lr_scheduler import fit

history = fit(model, train_loader, val_loader, optimizer, loss_fn, epochs=150)
print(history.best_val_acc, sum(history.polls), sum(history.optimizer_steps))
```

### API

| Object | Role |
|---|---|
| `EfficientPollingSGD` / `EfficientPollingOptimizer` | proposed method: polls on demand, with the divergence guard |
| `PollingSGD` / `PollingOptimizer` | base method: polls every batch |
| `SPSSGD` / `SPSOptimizer` | comparison baseline: stochastic Polyak step-size |
| `ArmijoSGD` / `ArmijoOptimizer` | comparison baseline: stochastic Armijo backtracking line search |
| `TRIGGERS` | the three polling triggers used in the ablation: `"backoff"` (default), `"fixed"`, `"random"` |
| `make_closure`, `accuracy`, `negative_loss` | batch closure and selection criteria (accuracy, or loss) |
| `StepInfo`, `EpochStats`, `History` | telemetry: chosen LR, polls, spikes, rollbacks, optimizer steps |
| `fit`, `train_epoch`, `evaluate` | optional training loop helpers, accepting a `torch.optim.lr_scheduler` for the plain-optimizer comparison methods |
| `StateSnapshot` | exact save/restore of parameters, buffers and optimizer state |

**Notes.** Despite the distribution name, these are **not** `torch.optim.lr_scheduler.LRScheduler` subclasses: they wrap the optimizer and are driven entirely through `optimizer.step(closure)`, so there is no separate `scheduler.step()` to call after it. Candidate learning rates are absolute and applied to every parameter group, overriding per-group learning rates. Pass `module=` (or the model itself as the first argument) whenever the forward pass mutates buffers, so trials cannot leak BatchNorm statistics. The closure must not call `backward()` or `zero_grad()` — the optimizer owns both.

---

## How it works

### Polling (replicated base method)

At each batch, after computing the gradient `g` from weights `θ`, every candidate learning rate is applied as a trial step and the winner is kept:

```
θ̂ₖ = θ − lrₖ · g               for each lrₖ ∈ C
k*  = argmax acc(θ̂ₖ, batch)     (ties favour the smallest lr)
θ   ← θ̂ₖ*
```

The candidate set is `C = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`, spanning four orders of magnitude around the base LR. The trial updates are realized by snapshotting the model + optimizer state once, then reloading it before each candidate step, so every candidate departs from an identical pre-step condition. Because the trials overwrite the weights, the winning step is reapplied once from the snapshot, so a poll costs `N + 1 = 6` optimizer steps rather than `N = 5`.

### Efficient Polling (proposed extension)

The selection mechanism is untouched, but a batch is polled only when needed:

1. **Adaptive polling schedule.** Let `K` be the poll interval. After a poll, if the selection is unchanged, `K ← min(2K, K_max)` (geometric backoff, capped at `K_max = 64`); if it changed, `K ← 1` (poll every batch until it stabilizes again). Between polls, a single blind SGD step uses the last selected LR. Backoff only engages once a poll has actually shown *signal* — i.e. the candidates' accuracies differ. At initialization every candidate ties, and treating a tie as a stable selection would stall training at the smallest candidate LR forever; the [trigger ablation](#trigger-ablation) reproduces exactly this failure in a variant that lacks the rule.

2. **Two-tier divergence guard.** Blind steps have no per-step validation, so a high-LR step can diverge. The guard reuses quantities already computed:
   - **Tier 2 — spike-triggered polls (prevention):** if the batch loss exceeds `γ · EMA(loss)` (`γ = 3`, `β = 0.9`), poll immediately so the accuracy criterion can reject an explosive step.
   - **Tier 1 — rollback checkpoints (recovery):** each poll snapshot doubles as a known-good checkpoint; if the loss is non-finite or exceeds `2·ln(C) ≈ 4.61`, restore the checkpoint and resume polling.

Across the five official runs, spike-triggered polls numbered 378 per run on average (6.6% of all polls) and the recovery tier fired only twice in the entire study, both in the same seed. Its necessity is real, though: an early unguarded run diverged to `NaN` at epoch 33 from a single blind step at `lr = 1e-1` and never recovered — the same failure that killed most of the hand-designed schedulers in the comparison (see [Results](#results)).

3. **Alternative triggers (ablation).** `trigger="fixed"` polls every `K + 1` batches at a constant interval instead of backing off, and `trigger="random"` polls each batch independently with probability `p`. Both keep rule 2 (the guard) and the selection mechanism unchanged, and both are calibrated to the ~5% poll rate the adaptive backoff measures, so a comparison against them isolates the trigger rather than the budget. See [Trigger ablation](#trigger-ablation).

| Symbol | Value | Role |
|---|---|---|
| `C` | `{1e-5, …, 1e-1}` | candidate learning rates |
| `lr_init` | `1e-3` | LR before the first poll |
| `K_max` | `64` | max poll interval (backoff cap) |
| `γ` | `3` | spike threshold (tier 2) |
| `β` | `0.9` | loss-EMA decay |
| `ℓ_rb` | `2·ln 10 ≈ 4.61` | rollback threshold (tier 1) |

---

## Results

Table below reproduces the paper's Table I: eleven configurations, each run on five seeds (42–46), 150 epochs, batch size 64 (704 batches/epoch, 105,600/run). Reported as mean ± sample standard deviation.

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

The three schedulers start at `1e-1` (the top of the candidate set) rather than at the baseline `1e-3`, since a decay schedule needs somewhere to decay from; SPS and Armijo are capped at that same `1e-1`, so no method may take a step the others were never allowed to consider. Despite that, cosine annealing, step decay and ReduceLROnPlateau diverge to `NaN` around epoch 28 on most of their seeds (5/5, 4/5 and 1/5 respectively) — the table still credits them with their best pre-divergence checkpoint, since every method is scored at its own best validation epoch. Both Polling variants hold that same `1e-1` for roughly thirty epochs across their 25 combined runs without a single failure: what breaks the schedulers is not the rate itself but the absence of a per-step check on it.

Both polling methods autonomously discover the same **two-phase schedule** entirely from batch-level feedback: the mean selected LR converges to `≈1e-1` within the first epoch, holds there for ~30 epochs, then collapses toward `≈1e-5` for fine refinement near convergence — base Polling completes the anneal between epochs 42–45, Efficient Polling more gradually, between epochs 49–78 (the interval hasn't reset to one on every batch yet).

| | |
|---|---|
| ![Loss curves](https://raw.githubusercontent.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/main/images/training_comparison_losses_all.png) | ![Learning-rate trajectories](https://raw.githubusercontent.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/main/images/training_comparison_LRs_all.png) |
| Training & validation loss, all eleven configurations. | Mean selected LR per epoch, symlog, with a cross marking divergence. |

![Polls per epoch](https://raw.githubusercontent.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/main/images/polls_per_epoch.png)

Polls concentrate exactly where the schedule changes: the steady-state floor is `704 / (K_max + 1) ≈ 11` polls/epoch, the median over all epochs is 16.4, and the count peaks at 231 in epoch 32 — the exact epoch the selected LR begins collapsing from `1e-1` to `1e-5`, where disagreeing polls keep resetting the interval to one. This is the mechanism that lets ~5% of the polls recover the full two-phase schedule.

### Cost model

A poll costs `N + 1 = 6` optimizer steps (one trial per candidate, plus reapplying the winner). With `P` polls out of `B = 105,600` batches:

```
S_eff = P·(N+1) + (B − P) = B + N·P
```

With the `P = 5,732.2` polls measured on average over the five runs, this gives `134,261` steps — a 79% reduction versus base Polling's `633,600`, exactly matching Table above. Only 26% of those steps come from polls; the remaining 74% are ordinary SGD steps, which is why the per-epoch wall-clock time sits at the level of plain SGD.

### Trigger ablation

Two control variants isolate the contribution of the adaptive backoff by replacing only the trigger, keeping the candidate set, selection rule and two-tier guard unchanged: `fixed` polls at a constant interval (`K = 19`) and `random` polls each batch independently with probability `p = 0.05`, both calibrated to the ~5% rate the backoff measures.

On final accuracy the **random trigger is competitive** — 84.02% test vs. 83.76% for the backoff, well inside the seed-to-seed spread. This is an honest negative result for the strong reading of the claim: at this budget, distributing polls uniformly at random is enough to track the schedule, provided the guard absorbs the cost of arriving late. The supported claim is the weaker one — the backoff reaches the same quality while spending its polls where they carry information (16.4 polls in a median epoch vs. a peak of 231 at the transition, a 14× ratio, against a flat ~40/epoch for both controls), needing **~2.4× fewer** spike-triggered guard interventions (378 vs. 919 and 888) and slightly fewer optimizer steps.

The **fixed-interval control exposes a real failure mode**: on 4/5 seeds it matches the other variants (84.17% ± 0.96% val), but on the remaining seed it never leaves the initialization plateau, ending at ~10% (random-guess) accuracy — because at initialization every candidate ties, the poll keeps returning the smallest candidate LR by the tie-break rule, and at `1e-5` the weights move too little to ever break the tie. The adaptive backoff is immune by construction, since it refuses to back off until a poll has actually discriminated between candidates (rule 1 above).

---

## Presentation

🎥 [Watch the presentation video](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/videos/apresentação.mp4) · 📊 [Slides (PDF)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/apresentacao_polling.pdf) · [Slides (PPTX)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/apresentacao_polling.pptx) · 📄 [Paper (LaTeX source)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/main.tex)

---

## Repository structure

```
.
├── src/efficient_polling_lr_scheduler/     # the installable package
│   ├── polling.py             # base method (Tan et al.)
│   ├── efficient.py           # Efficient Polling (ours), incl. the fixed/random triggers
│   ├── baselines.py           # SPS and Armijo backtracking comparison optimizers
│   ├── _snapshot.py           # exact state save/restore for trial steps
│   ├── closures.py            # batch closures and selection criteria
│   └── training.py            # optional fit/train_epoch/evaluate helpers
├── tests/                     # pytest suite for the algorithms
├── examples/
│   ├── cifar10.py             # reproduces all eleven configurations from the CLI
│   └── plot_results.py        # redraws the figures from the recorded runs
├── notebooks/
│   └── cifar10.ipynb          # original experiments: data, model, all 11 methods, plots
├── docs/
│   ├── main.tex                # the paper (IEEE format)
│   ├── apresentacao_polling.pdf
│   └── apresentacao_polling.pptx
├── videos/
│   └── apresentação.mp4       # presentation video
├── images/                    # figures used in the paper and this README
├── results/cifar10/           # the 55 recorded runs (11 methods × 5 seeds) behind the paper
├── models/                    # best checkpoints per method (.pt, gitignored)
├── pyproject.toml
├── CHANGELOG.md
├── README.md
└── README(pt-br).md
```

## Setup

To *use* the methods, all you need is the package (Python 3.10+, PyTorch 2.0+):

```bash
pip install efficient-polling-lr-scheduler
```

To *reproduce the experiments*, clone the repository and install with the extras. A CUDA-capable GPU is recommended (CPU works but is slow):

```bash
git clone https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks.git
cd Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,examples]" jupyter
```

### Dataset

The experiments load the **CIFAR-10 Python** version from a local directory (the pickled `data_batch_*` / `test_batch` files). Download it from the [official site](https://www.cs.toronto.edu/~kriz/cifar.html):

```bash
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
```

## Running

The example script runs every configuration over one or more seeds and prints the comparison table, resuming a sweep from any results already on disk:

```bash
python examples/cifar10.py --data-dir /path/to/cifar-10-batches-py --seeds 42 43 44 45 46
# one method, one seed, shorter run:
python examples/cifar10.py --data-dir ... --methods efficient --seeds 42 --epochs 20
```

Results are written to `--results-dir` (default `results/cifar10/`) as one JSON file per `(method, seed)`. `examples/plot_results.py` redraws the figures from those files, so a plot can never disagree with the table:

```bash
python examples/plot_results.py --results-dir results/cifar10 --out-dir images
```

Run the test suite with `pytest`.

Alternatively, open the original notebook and run the cells top to bottom, pointing `DATA_DIR` (in the **Constants** cell) at the extracted `cifar-10-batches-py` directory:

```bash
jupyter notebook notebooks/cifar10.ipynb
```

The notebook is organized as: Imports → Constants → Configs (seeds `42`–`46`, device) → Data (dataset, normalization stats, 90/10 train/val split) → Model (`SimpleCIFAR10CNN`, ~0.56M params) → Train (all eleven configurations, one shared loop) → Animations & plots → Test. Best checkpoints are written to `models/`.

> **Reproducibility.** Five seeds (42–46) each fix weight init, data shuffling and the train/val split, so on a given seed every method starts from the same weights and sees the same batch order. All numbers above are the mean ± sample standard deviation over the five runs.

---

## Experimental setup

- **Dataset:** CIFAR-10 — 45,000 train / 5,000 val / 10,000 test, normalized per channel with training statistics.
- **Model:** `SimpleCIFAR10CNN`, a 5-layer CNN (64→64→128→128→256 conv channels, `3×3` kernels, ReLU, MaxPool, AdaptiveAvgPool, Linear head), **557,898 parameters**, no batch norm or dropout so the optimizer is the only source of adaptation.
- **Optimizer:** vanilla SGD (no momentum, no weight decay) for the proposed and replicated methods, batch size 64, base LR `1e-3`, 150 epochs (704 batches/epoch, 105,600 total).
- **Comparison methods:** Adam, cosine annealing, step decay, ReduceLROnPlateau, SPS (Polyak step-size), Armijo backtracking line search, and the replicated base Polling method — eight in total, plus two trigger-ablation variants of the proposed method.
- **Seeds:** five (42–46) per configuration, eleven configurations, 55 runs total.
- **Hardware:** single NVIDIA GeForce RTX 5070 (12 GB).

---

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{souzasilva_efficient_polling_lr_scheduler,
  title  = {Efficient Polling-Based Learning Rate Optimization for Neural Networks},
  author = {de Souza Silva, Jos{\'e} R. and B. A. da Silva, Luiz Henrique and B. de Souza, Caio B. and Balieiro, Andson M.},
  year   = {2026},
  note   = {Centro de Inform{\'a}tica (CIn), Universidade Federal de Pernambuco (UFPE)},
  url    = {https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks}
}
```

The base Polling method is from Tan et al. (see `docs/base_paper.pdf`). The full paper, with related work, the cost model derivation and the trigger ablation, is at `docs/main.tex`.

To cite the software specifically, add `note = {Python package \texttt{efficient-polling-lr-scheduler}}` or reference [the PyPI project](https://pypi.org/project/efficient-polling-lr-scheduler/).

## 🧑‍💻 Authors

| [<img src="https://github.com/luiz-linkezio.png" width=115><br><sub>Luiz Henrique</sub><br>](https://github.com/luiz-linkezio) <sub>Developer</sub><br> <sub>[LinkedIn](https://www.linkedin.com/in/lhbas/)</sub><br> <sub>Portfolio</sub> | [<img src="https://github.com/dev-joseronaldo.png" width=115><br><sub>José Ronaldo</sub><br>](https://github.com/Dev-JoseRonaldo) <sub>Developer</sub><br> <sub>[LinkedIn](https://www.linkedin.com/in/devjoseronaldo/)</sub><br> <sub>[Portfolio](https://joseronaldo.netlify.app/)</sub> |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |

Universidade Federal de Pernambuco, Recife, Brazil. The paper additionally credits Caio B. B. de Souza (UPE) and Andson M. Balieiro (CIn/UFPE) — see [Citation](#citation).

## License

MIT — see the [LICENSE](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/LICENSE) file in this repository.
