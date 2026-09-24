# Efficient Polling-Based Learning Rate Optimization for Neural Networks

> Choose the learning rate by measuring it on the batch, at close to the cost of plain SGD.

[![PyPI](https://img.shields.io/pypi/v/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![Python](https://img.shields.io/pypi/pyversions/efficient-polling-lr-scheduler.svg)](https://pypi.org/project/efficient-polling-lr-scheduler/)
[![License](https://img.shields.io/pypi/l/efficient-polling-lr-scheduler.svg)](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/LICENSE)

```bash
pip install efficient-polling-lr-scheduler
```

[🇧🇷 Versão em português](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/README(pt-br).md) · [How it works](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/methods.md) · [Results](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/results.md) · [Reproducing the experiments](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/reproducing.md)

The **Polling Method** of [Tan et al.](https://doi.org/10.3390/jeta4010001) picks the learning rate by trying several candidates on every batch and keeping the one that most improves batch accuracy. It works, and it multiplies the cost of training by the number of candidates. This repository replicates it and adds two extensions that keep the selection and drop most of the cost:

- **Efficient Polling** polls on demand. An exponential backoff stretches the gap between polls while the choice is stable, and a two-tier divergence guard protects the steps in between.
- **Efficient Relative Polling** drops the fixed grid of candidates. The user picks one learning rate and one multiplier, and each poll tries the rate in use and its two neighbours.

Both ship as a PyTorch package. They are benchmarked against eight comparison methods and two ablations on five datasets, five seeds each.

## Results at a glance

Test accuracy, mean over five seeds of 150 epochs, with the same hyperparameters on every dataset. The last row is the best of the other comparison methods on each dataset: fixed-rate SGD, Adam, the three schedulers, SPS and Armijo.

| Method | CIFAR-10 | CIFAR-100 | Fashion-MNIST | MNIST | Covertype | Optimizer steps, × SGD |
|---|---|---|---|---|---|---|
| Polling (base paper) | 83.83% | 52.99% | 91.83% | 98.14% | 75.15% | 6.00 |
| Efficient Polling (ours) | 83.76% | 25.07% | 91.84% | 98.82% | **75.79%** | 1.18–2.23 |
| Efficient Relative Polling (ours) | **84.11%** | **53.05%** | 91.70% | 98.59% | 75.41% | 1.04–1.21 |
| Best comparison method | 84.09%, Armijo | 53.03%, plateau | **92.70%**, Adam | **99.45%**, Adam | 75.64%, cosine | 1.00 |

- **Efficient Relative Polling matches the base method everywhere at a fraction of its cost.** It is above base Polling on four datasets and within one standard deviation of it on Fashion-MNIST, with at most 1.21× the optimizer steps of plain SGD where base Polling takes 6×. It is the best method on CIFAR-10 and CIFAR-100.
- **No method wins everywhere.** Adam leads on MNIST and Fashion-MNIST, where the polling methods trail it by 0.6 to 1.3 points.
- **Efficient Polling can stall at its smallest candidate.** When batch accuracy at chance rarely tells the candidates apart, the tie-break keeps choosing `1e-5` and the backoff reads the repetition as a stable choice. On CIFAR-100 every seed lost epochs this way, and two never trained.

[Results](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/results.md) has the full tables and figures, the trigger ablation, the initial-rate robustness runs and the mechanism behind the stall.

## Quickstart

Polling needs to re-evaluate the model to score a candidate step, so instead of the bare `optimizer.step()` you pass a **closure** that returns `(loss, score)`: the same contract as `torch.optim.LBFGS`, plus the score to maximize. `make_closure` builds it for you:

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

No learning-rate schedule, no warmup, no tuning: the learning rate is *measured*. `PollingSGD` gives the base method, which polls every batch. Efficient Relative Polling takes one rate instead of a grid, and either extension can wrap an optimizer you already have:

```python
from efficient_polling_lr_scheduler import EfficientPollingOptimizer, EfficientRelativePollingSGD

# one rate and one multiplier instead of a fixed grid
optimizer = EfficientRelativePollingSGD(model, lr=1e-3, multiplier=10.0, lr_max=1e-1)

# or Efficient Polling around any optimizer
optimizer = EfficientPollingOptimizer(
    torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9),
    candidate_lrs=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    module=model,          # so BatchNorm buffers are restored between trials
    max_poll_interval=64,  # backoff cap; 0 polls every batch
)
```

The package also ships the two comparison baselines that derive the step size from the current batch, the stochastic Polyak step-size (SPS) and Armijo backtracking line search, and optional training helpers that accept a plain optimizer and a `torch.optim.lr_scheduler` too, so every method goes through the same loop:

```python
from efficient_polling_lr_scheduler import SPSSGD, ArmijoSGD, fit

optimizer = SPSSGD(model, lr=1e-3, max_lr=0.1)  # or ArmijoSGD(model, lr=1e-3, lr_max=0.1)

history = fit(model, train_loader, val_loader, optimizer, loss_fn, epochs=150)
print(history.best_val_acc, sum(history.polls), sum(history.optimizer_steps))
```

### API

| Object | Role |
|---|---|
| `EfficientPollingSGD` / `EfficientPollingOptimizer` | Efficient Polling: polls on demand, with the divergence guard |
| `PollingSGD` / `PollingOptimizer` | base method: polls every batch |
| `EfficientRelativePollingSGD` / `EfficientRelativePollingOptimizer` | Efficient Relative Polling (experimental): three candidates around the rate in use, uncapped TCP-style backoff, restarts from the best point |
| `EfficientRelativeEpochPolling` | the same at epoch granularity, driven by `fit(..., epoch_polling=...)` over a plain optimizer |
| `EfficientRelativeNarrowingPollingSGD` / `EfficientRelativeNarrowingPollingOptimizer` | Efficient Relative Narrowing Polling (experimental, no results yet): Efficient Relative Polling with a jump that narrows between two rates when the rate stays bracketed for a while, and widens back when it keeps moving one way |
| `SPSSGD` / `SPSOptimizer` | comparison baseline: stochastic Polyak step-size |
| `ArmijoSGD` / `ArmijoOptimizer` | comparison baseline: stochastic Armijo backtracking line search |
| `TRIGGERS` | the three polling triggers used in the ablation: `"backoff"` (default), `"fixed"`, `"random"` |
| `make_closure`, `accuracy`, `negative_loss` | batch closure and selection criteria (accuracy, or loss) |
| `StepInfo`, `EpochStats`, `History` | telemetry: chosen LR, polls, spikes, rollbacks, optimizer steps |
| `fit`, `train_epoch`, `evaluate` | optional training loop helpers, accepting a `torch.optim.lr_scheduler` for the plain-optimizer comparison methods |
| `StateSnapshot` | exact save/restore of parameters, buffers and optimizer state |

**Notes.** Despite the distribution name, these are **not** `torch.optim.lr_scheduler.LRScheduler` subclasses: they wrap the optimizer and are driven entirely through `optimizer.step(closure)`, so there is no separate `scheduler.step()` to call after it. Candidate learning rates are absolute and applied to every parameter group, overriding per-group learning rates. Pass `module=` (or the model itself as the first argument) whenever the forward pass mutates buffers, so trials cannot leak BatchNorm statistics. The closure must not call `backward()` or `zero_grad()`: the optimizer owns both.

## Documentation

- [How it works](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/methods.md): the three methods, their rules and hyperparameters, and the cost model.
- [Results](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/results.md): every table and figure on the five datasets, the trigger ablation and the initial-rate robustness runs.
- [Reproducing the experiments](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/docs/reproducing.md): setup, datasets, the command line and the notebook, and how the repository is laid out.

## Citation

If you use this work, please cite:

```bibtex
@misc{souzasilva_efficient_polling_lr_scheduler,
  title  = {Efficient Polling-Based Learning Rate Optimization for Neural Networks},
  author = {de Souza Silva, Jos{\'e} R. and B. A. da Silva, Luiz Henrique and B. de Souza, Caio B. and Balieiro, Andson M.},
  year   = {2026},
  note   = {Centro de Inform{\'a}tica (CIn), Universidade Federal de Pernambuco (UFPE)},
  url    = {https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks}
}
```

The base Polling method is from Tan, Choong and Lau, [Expediting Convergence via Polling Optimisation for Gradient Descent in Neural Networks](https://doi.org/10.3390/jeta4010001) (2025). To cite the software specifically, add `note = {Python package \texttt{efficient-polling-lr-scheduler}}` or reference [the PyPI project](https://pypi.org/project/efficient-polling-lr-scheduler/).

## 🧑‍💻 Authors

| [<img src="https://github.com/luiz-linkezio.png" width=115><br><sub>Luiz Henrique</sub><br>](https://github.com/luiz-linkezio) <sub>Developer</sub><br> <sub>[LinkedIn](https://www.linkedin.com/in/lhbas/)</sub><br> <sub>Portfolio</sub> | [<img src="https://github.com/dev-joseronaldo.png" width=115><br><sub>José Ronaldo</sub><br>](https://github.com/Dev-JoseRonaldo) <sub>Developer</sub><br> <sub>[LinkedIn](https://www.linkedin.com/in/devjoseronaldo/)</sub><br> <sub>[Portfolio](https://joseronaldo.netlify.app/)</sub> |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |

Universidade Federal de Pernambuco, Recife, Brazil. The paper additionally credits Caio B. B. de Souza (UPE) and Andson M. Balieiro (CIn/UFPE); see [Citation](#citation).

## License

MIT, see the [LICENSE](https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/blob/main/LICENSE) file in this repository.
