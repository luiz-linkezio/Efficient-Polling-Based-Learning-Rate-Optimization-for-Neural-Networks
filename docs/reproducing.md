# Reproducing the experiments

[🇧🇷 Português](pt-br/reproducing.md) · [README](../README.md) · [How it works](methods.md) · [Results](results.md)

- [Setup](#setup)
- [Datasets](#datasets)
- [Running](#running): from the command line or from the notebook
- [Experimental setup](#experimental-setup)
- [Repository layout](#repository-layout)

## Setup

To *use* the methods, all you need is the package (Python 3.10+, PyTorch 2.0+):

```bash
pip install efficient-polling-lr-scheduler
```

To *reproduce the experiments*, clone the repository and install it with the extras. A CUDA-capable GPU is recommended (CPU works but is slow):

```bash
git clone https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks.git
cd Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,benchmark]" jupyter
nbstripout --install
```

The last line makes git strip the notebook's outputs whenever it is committed, so the logs and animations of a run never land in the history; CI rejects a notebook committed with outputs. Run the test suite with `pytest`.

## Datasets

Every dataset is read straight from the files its authors publish: no torchvision, no download step hidden inside a run. Four are images; **Covertype is not**. It has 581,012 rows of 54 cartographic features and seven forest cover types, which swaps the CNN for an MLP and takes the comparison out of vision altogether:

| Dataset | Input | Classes | The data directory should hold | Source |
|---|---|---|---|---|
| `cifar10` (default) | 32×32 colour | 10 | `data_batch_1`…`data_batch_5`, `test_batch` | [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html) |
| `cifar100` | 32×32 colour | 100 | `train`, `test` (fine labels) | [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html) |
| `mnist` | 28×28 grey | 10 | the four IDX files | [ossci mirror](https://ossci-datasets.s3.amazonaws.com/mnist/) |
| `fashion_mnist` | 28×28 grey | 10 | the four IDX files | [zalandoresearch](https://github.com/zalandoresearch/fashion-mnist) |
| `covertype` | 54 features | 7 | `covtype.data` (or `.gz`) | [UCI](https://archive.ics.uci.edu/dataset/31/covertype) |

```bash
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
```

The IDX files are read compressed or not, dashed (`train-images-idx3-ubyte`) or dotted (`train-images.idx3-ubyte`), so there is nothing to unpack or rename. Covertype follows the published protocol: the first 15,120 rows are what the sweep fits on (its own 90/10 train/val split runs inside them), the remaining 565,892 are the test set. Images are normalized per channel and Covertype per feature, always with statistics of the training split. Covertype's 44 wilderness-area and soil-type columns are 0/1 flags and stay on that scale: z-scoring a flag set in a handful of rows turns it into a value in the hundreds, and two soil types never occur in the 15,120 training rows at all, which with a plain z-score would send a test row that has one set off to 1e8.

## Running

Runs are recorded in `results/<dataset>/`, one JSON file per `(method, seed)`, and the best checkpoint of each goes to `models/`. A run that already has a record is read back instead of retrained, so a sweep can be stopped and resumed at any point, from either entry point. Both run the same code, the `benchmark` package.

### From the command line

Run from the repository root:

```bash
python -m benchmark --data-dir /path/to/cifar-10-batches-py --seeds 42 43 44 45 46
# one method, one seed, a shorter run:
python -m benchmark --data-dir ... --methods efficient --seeds 42 --epochs 20
# another dataset, with the ablations calibrated the way its recorded runs were:
python -m benchmark --dataset mnist --data-dir /path/to/MNIST --seeds 42 43 44 45 46 --calibrate-ablations
# the initial-rate robustness runs, recorded as <method>_lr<rate>:
python -m benchmark --data-dir ... --methods efficient efficient_relative --seeds 42 --lr 1e-5
```

The sweep goes seed by seed and prints the table at the end. Every hyperparameter has a flag, and `python -m benchmark --help` lists them with the values the recorded runs used. `--calibrate-ablations` sets both trigger ablations to the poll rate `efficient` measured on the first seed, so that run has to exist first; in the default method order it does.

The figures are drawn from the records, so a figure can never disagree with the table:

```bash
python -m benchmark.plots                   # CIFAR-10, into images/cifar10/
python -m benchmark.plots --dataset mnist   # into images/mnist/
```

### From the notebook

```bash
jupyter notebook notebooks/benchmark.ipynb
```

Set `DATASET` and its entry in `DATA_DIRS` in the first code cells, then run the cells top to bottom: Setup, Hyperparameters, Data, Model, Training runs (one cell per family of methods), Figures and animations, Test. The notebook builds an `Experiment` from the `benchmark` package and calls it, so what it runs is exactly what the command line runs.

> **Reproducibility.** Five seeds (42–46) each fix weight init, data shuffling and the train/val split, so on a given seed every method starts from the same weights and sees the same batch order.

## Experimental setup

- **Datasets:** CIFAR-10 and CIFAR-100, 45,000 train / 5,000 val / 10,000 test. MNIST and Fashion-MNIST, 54,000 / 6,000 / 10,000. Covertype, 13,608 / 1,512 / 565,892.
- **Models:** `SimpleCNN` on the image datasets, a 5-layer CNN (64→64→128→128→256 conv channels, `3×3` kernels, ReLU, MaxPool, AdaptiveAvgPool, Linear head) with **557,898 parameters** on CIFAR-10, 581,028 on CIFAR-100 and 556,746 on MNIST and Fashion-MNIST. Only the first convolution and the classifier change with the dataset; the global pool means the input side length never enters. Covertype gets `SimpleMLP` (512→256→128, 193,287 parameters). Neither has batch norm or dropout, so the optimizer is the only source of adaptation.
- **Optimizer:** vanilla SGD (no momentum, no weight decay) for the proposed and replicated methods, batch size 64, base LR `1e-3`, 150 epochs.
- **Comparison methods:** Adam, cosine annealing, step decay, ReduceLROnPlateau, SPS (Polyak step-size), Armijo backtracking line search and the replicated base Polling method, eight in total, plus two trigger-ablation variants of Efficient Polling. The three schedulers start at `1e-1`, and SPS and Armijo are capped there.
- **Seeds:** five (42–46) per configuration: thirteen configurations, 65 runs per dataset and 325 in all, plus four initial-rate robustness runs on CIFAR-10, seed 42.
- **Hardware:** a single NVIDIA GeForce RTX 5070 (12 GB). CIFAR-100, Fashion-MNIST, MNIST and Covertype ran three at a time on it, so their timings are not reported.

## Repository layout

```
.
├── src/efficient_polling_lr_scheduler/   # the package published on PyPI
│   ├── polling.py             # base method (Tan et al.)
│   ├── efficient.py           # Efficient Polling, incl. the fixed/random triggers
│   ├── efficient_relative.py  # Efficient Relative Polling, per batch and per epoch
│   ├── baselines.py           # SPS and Armijo backtracking comparison optimizers
│   ├── _snapshot.py           # exact state save/restore for trial steps
│   ├── closures.py            # batch closures and selection criteria
│   └── training.py            # optional fit/train_epoch/evaluate helpers
├── benchmark/                 # the experiment, run by python -m benchmark and the notebook
│   ├── datasets.py            # the five dataset readers, no torchvision
│   ├── models.py              # SimpleCNN and SimpleMLP
│   ├── methods.py             # the thirteen configurations and their hyperparameters
│   ├── sweep.py               # runs configurations over seeds, records and reads back runs
│   └── plots.py               # the figures, drawn from the records
├── notebooks/benchmark.ipynb  # drives the benchmark package, committed without outputs
├── tests/                     # pytest suite for the package and the benchmark
├── results/<dataset>/         # one JSON per (method, seed)
├── images/<dataset>/          # the figures of each dataset
├── docs/                      # how it works, results, reproducing; pt-br/ holds the Portuguese
├── models/                    # best checkpoints (.pt, gitignored)
├── pyproject.toml
├── CHANGELOG.md
├── README.md
└── README(pt-br).md
```

To add a method, subclass `PollingOptimizer` in `src/`. It inherits exact snapshot and restore, the optimizer protocol and the `StepInfo` telemetry that `fit()` aggregates; set `optimizer_steps` to what the batch actually cost and the cost model stays honest. Then add an entry to `LABELS` and a branch to `build_optimizer()` in `benchmark/methods.py`, and the method joins the sweep, the tables and the figures.
