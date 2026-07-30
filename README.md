# Efficient Polling-Based Learning Rate Optimization for Neural Networks

> Get the accuracy of polling-based learning-rate selection at essentially the cost of plain SGD.

[![PyPI](https://img.shields.io/pypi/v/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![Python](https://img.shields.io/pypi/pyversions/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![License](https://img.shields.io/pypi/l/efficient-polling-lr-scheduler.svg)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/LICENSE)

```bash
pip install efficient-polling-lr-scheduler
```

[🇧🇷 Versão em português](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/README(pt-br).md) · [🎥 Presentation video](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/videos/apresentação.mp4)

This repository replicates the **Polling Method** of Tan et al. on CIFAR-10 and introduces **Efficient Polling**, a novel extension that recovers the same learning-rate schedule — and the same accuracy — while polling only **5% of batches**, cutting optimizer steps by 75% and per-epoch wall-clock time by **3.3×**. Both methods ship as a PyTorch package.

---

## TL;DR

The learning rate is the single most influential hyperparameter in gradient-based training. Instead of picking it by hand or by a fixed schedule, **polling** tests several candidate learning rates at every batch and keeps the one that most improves batch accuracy. It works remarkably well, but it triples training time.

**Efficient Polling** observes that the polled choice is highly redundant — within each training phase consecutive polls pick the same learning rate — and polls *on demand* instead: an exponential-backoff schedule doubles the gap between polls while the selection is stable, and a two-tier divergence guard protects the unpolled steps.

| Method | Best Val | Test Acc | Test Loss | Polled Batches | s/Epoch |
|---|---|---|---|---|---|
| Baseline (fixed SGD, `1e-3`) | 57.28% | 56.70% | 1.2155 | — | 2.50 |
| Polling (base paper) | 84.65% | **83.99%** | **0.6832** | 100% | 9.10 |
| **Efficient Polling (ours)** | **85.07%** | 83.93% | 0.7319 | **5.05%** | **2.79** |

*150 epochs, single seed (42), one fully reproducible run per method, NVIDIA RTX 5070.*

Efficient Polling **matches** the base method's accuracy (within 0.1 pp on test) at only **12% over plain SGD** — versus the base method's +264%.

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

The optional training helpers run a full comparison in a few lines, and accept a plain optimizer too — so the baseline goes through the same loop:

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
| `make_closure`, `accuracy`, `negative_loss` | batch closure and selection criteria (accuracy, or loss) |
| `StepInfo`, `EpochStats`, `History` | telemetry: chosen LR, polls, spikes, rollbacks, optimizer steps |
| `fit`, `train_epoch`, `evaluate` | optional training loop helpers |
| `StateSnapshot` | exact save/restore of parameters, buffers and optimizer state |

**Notes.** Despite the distribution name, these are **not** `torch.optim.lr_scheduler.LRScheduler` subclasses: they wrap the optimizer and are driven entirely through `optimizer.step(closure)`, so there is no separate `scheduler.step()` to call after it. Candidate learning rates are absolute and applied to every parameter group, overriding per-group learning rates. Pass `module=` (or the model itself as the first argument) whenever the forward pass mutates buffers, so trials cannot leak BatchNorm statistics. The closure must not call `backward()` or `zero_grad()` — the optimizer owns both.

---

## How it works

### Polling (replicated base method)

At each batch, after computing the gradient `g` from weights `θ`, every candidate learning rate is applied as a trial step and the winner is kept:

```
ĝθₖ = θ − lrₖ · g           for each lrₖ ∈ C
k*  = argmax acc(ĝθₖ, batch)   (ties favour the smallest lr)
θ   ← ĝθₖ*
```

The candidate set is `C = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`, spanning five orders of magnitude around the base LR. The trial updates are realized by snapshotting the model + optimizer state once, then reloading it before each candidate step, so every candidate departs from an identical pre-step condition. This costs `N = 5` trial updates **per batch**.

### Efficient Polling (proposed extension)

The selection mechanism is untouched, but a batch is polled only when needed:

1. **Adaptive polling schedule.** Let `K` be the poll interval. After a poll, if the selection is unchanged, `K ← min(2K, K_max)` (geometric backoff, capped at `K_max = 64`); if it changed, `K ← 1` (poll every batch until it stabilizes again). Between polls, a single blind SGD step uses the last selected LR.

2. **Two-tier divergence guard.** Blind steps have no per-step validation, so a high-LR step can diverge. The guard reuses quantities already computed:
   - **Tier 2 — spike-triggered polls (prevention):** if the batch loss exceeds `γ · EMA(loss)` (`γ = 3`, `β = 0.9`), poll immediately so the accuracy criterion can reject an explosive step.
   - **Tier 1 — rollback checkpoints (recovery):** each poll snapshot doubles as a known-good checkpoint; if the loss is non-finite or exceeds `2·ln(C) ≈ 4.61`, restore the checkpoint and resume polling.

In the official run, the spike tier alone was sufficient — **0 rollbacks** were ever triggered. Its necessity is real, though: an early unguarded run diverged to `NaN` at epoch 33 from a single blind step at `lr = 1e-1` and never recovered.

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

Both polling methods autonomously discover a **two-phase schedule** entirely from batch-level feedback: the highest candidate (`≈ 1e-1`) drives rapid loss reduction for the first ~36 epochs, then the selection collapses to the smallest candidate (`≈ 1e-5`) for fine refinement near convergence. Efficient Polling recovers the same schedule while polling a tiny fraction of batches.

| | |
|---|---|
| ![Loss curves](https://raw.githubusercontent.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/main/images/training_comparison_losses.png) | ![Learning-rate trajectories](https://raw.githubusercontent.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/main/images/training_comparison_LRs.png) |
| Training & validation loss over 150 epochs. | Mean selected LR per epoch (symlog). |

![Polls per epoch](https://raw.githubusercontent.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/main/images/polls_per_epoch.png)

Polls concentrate exactly where the schedule changes: outside the phase transition the count sits at the steady-state floor of `704 / K_max ≈ 11` polls/epoch; it spikes to 233 at epoch 33 — the exact moment the selected LR collapses from `1e-1` to `1e-5` — when disagreeing polls keep resetting the interval to one. This is the mechanism that lets 5% of the polls recover the full schedule.

### Cost model

With `P = 5,337` polls over `B = 105,600` batches, each poll costing `N + 1 = 6` steps and each unpolled batch costing 1:

```
S_eff = P·(N+1) + (B − P) = 5,337·6 + 100,263 = 132,285 optimizer steps
```

versus `528,000` for base Polling — a 75% reduction, exactly reproducing the measured step count.

---

## Presentation

🎥 [Watch the presentation video](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/videos/apresentação.mp4) · 📊 [Slides (PDF)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/apresentacao_polling.pdf) · [Slides (PPTX)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/apresentacao_polling.pptx)

---

## Repository structure

```
.
├── src/efficient_polling_lr_scheduler/     # the installable package
│   ├── polling.py             # base method (Tan et al.)
│   ├── efficient.py           # Efficient Polling (ours)
│   ├── _snapshot.py           # exact state save/restore for trial steps
│   ├── closures.py            # batch closures and selection criteria
│   └── training.py            # optional fit/train_epoch/evaluate helpers
├── tests/                     # pytest suite for the algorithms
├── examples/
│   └── cifar10.py             # reproduces the paper's three runs from the CLI
├── notebooks/
│   └── cifar10.ipynb          # original experiments: data, model, all 3 methods, plots
├── docs/
│   ├── apresentacao_polling.pdf
│   └── apresentacao_polling.pptx
├── videos/
│   └── apresentação.mp4       # presentation video
├── images/                    # figures used in the paper and this README
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

The example script runs all three methods and prints the comparison table:

```bash
python examples/cifar10.py --data-dir /path/to/cifar-10-batches-py
# one method only, shorter run:
python examples/cifar10.py --data-dir ... --methods efficient --epochs 20
```

Run the test suite with `pytest`.

Alternatively, open the original notebook and run the cells top to bottom, pointing `DATA_DIR` (in the **Constants** cell) at the extracted `cifar-10-batches-py` directory:

```bash
jupyter notebook notebooks/cifar10.ipynb
```

The notebook is organized as: Imports → Constants → Configs (seed `42`, device) → Data (dataset, normalization stats, 90/10 train/val split) → Model (`SimpleCIFAR10CNN`, ~0.56M params) → Train (Baseline, Polling, Efficient Polling) → Animations & plots → Test. Best checkpoints are written to `models/`.

> **Reproducibility.** A single seed (42) fixes weight init, data shuffling, and the train/val split, so the three methods differ only in their learning-rate logic. All numbers above come from one run per method.

---

## Experimental setup

- **Dataset:** CIFAR-10 — 45,000 train / 5,000 val / 10,000 test, normalized per channel with training statistics.
- **Model:** `SimpleCIFAR10CNN`, a 5-layer CNN (64→64→128→128→256 conv channels, `3×3` kernels, ReLU, MaxPool, AdaptiveAvgPool, Linear head), **557,898 parameters**, no batch norm or dropout so the optimizer is the only source of adaptation.
- **Optimizer:** vanilla SGD (no momentum, no weight decay), batch size 64, base LR `1e-3`, 150 epochs (704 batches/epoch, 105,600 total).
- **Hardware:** single NVIDIA GeForce RTX 5070 (12 GB).

---

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{henrique_efficient_polling_lr_scheduler,
  title  = {Efficient Polling-Based Learning Rate Optimization for Neural Networks},
  author = {Henrique, Luiz and Ronaldo, Jos{\'e}},
  year   = {2026},
  note   = {Universidade Federal de Pernambuco},
  url    = {https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks}
}
```

The base Polling method is from Tan et al. (see `docs/base_paper.pdf`).

To cite the software specifically, add `note = {Python package \texttt{efficient-polling-lr-scheduler}}` or reference [the PyPI project](https://pypi.org/project/efficient-polling-lr-scheduler/).

## 🧑‍💻 Authors

| [<img src="https://github.com/luiz-linkezio.png" width=115><br><sub>Luiz Henrique</sub><br>](https://github.com/luiz-linkezio) <sub>Developer</sub><br> <sub>[LinkedIn](https://www.linkedin.com/in/lhbas/)</sub><br> <sub>Portfolio</sub> | [<img src="https://github.com/dev-joseronaldo.png" width=115><br><sub>José Ronaldo</sub><br>](https://github.com/Dev-JoseRonaldo) <sub>Developer</sub><br> <sub>[LinkedIn](https://www.linkedin.com/in/devjoseronaldo/)</sub><br> <sub>[Portfolio](https://joseronaldo.netlify.app/)</sub> |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |

Universidade Federal de Pernambuco, Recife, Brazil.

## License

MIT — see the [LICENSE](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/LICENSE) file in this repository.
