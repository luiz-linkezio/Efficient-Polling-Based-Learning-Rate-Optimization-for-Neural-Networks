# Reproducing the experiments

[🇧🇷 Português](pt-br/reproducing.md) · [README](../README.md) · [How it works](methods.md) · [Results](results.md)

- [Setup](#setup)
- [Datasets](#datasets)
- [Running](#running): from the command line or from the notebook
- [Learning-rate rounds](#learning-rate-rounds): every method on SGD or on Adam, from one rate, one table per rate
- [Running on a SLURM cluster](#running-on-a-slurm-cluster): the whole study in one job, several runs per GPU
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
# a learning-rate round, recorded in results/<dataset>/rounds/adam_lr1/:
python -m benchmark --data-dir ... --seeds 42 43 44 45 46 --round adam:1
# SPS and Armijo under a 1e-7 ceiling, recorded in results/<dataset>/ceilings/sgd_lr1e-07/:
python -m benchmark --data-dir ... --seeds 42 43 44 45 46 --ceiling 1e-7
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

## Learning-rate rounds

The main table fixes the base optimizer at SGD and the starting rate at `1e-3`, and varies the method. A round fixes both at other values: every method steps with one base optimizer and is given one learning rate, so a round shows how each method copes with a rate chosen too high or too low, without mixing the optimizer into the comparison. There are six, SGD and Adam each from `1`, `1e-3` and `1e-7`, defined in `benchmark/rounds.py`. Each method takes the round's rate wherever it asks for one:

| Method | What the round's rate is |
|---|---|
| Fixed rate | the rate of the whole run, on the round's optimizer |
| Cosine annealing, step decay, ReduceLROnPlateau | where the schedule starts, instead of the main table's `1e-1` |
| Polling, Efficient Polling | the centre of the candidate grid, two decades either side, so from `1` the grid reaches `1e+2`, as in the initial-rate runs |
| Trigger ablations | as for Efficient Polling, with the poll rate calibrated to the round's own Efficient Polling run |
| Efficient Relative Polling, per batch and per epoch | where it starts, under the main table's `1e-1` ceiling, which the rounds from `1` raise to `1` because the method refuses to start above its ceiling |
| Efficient Relative Narrowing Polling | as Efficient Relative Polling per batch, under the same ceiling |

Adam, SPS and Armijo sit the rounds out. Adam is the base optimizer of half of them. SPS and Armijo never read a starting rate: the Polyak step overwrites it on the first batch, and the line search starts every batch from its ceiling. Their own test moves that ceiling through the same three rates instead, on SGD only, because both formulas assume the step follows the gradient, which Adam's does not.

A round records into `results/<dataset>/rounds/<optimizer>_lr<rate>/` and the ceiling test into `results/<dataset>/ceilings/sgd_lr<rate>/`, with the checkpoints in the same folders under `models/`, so neither mixes with the main table.

The six rounds and the three ceilings are one study, run by one command and read back as **one table per initial rate**: every round method on SGD, then SPS and Armijo with that rate as their ceiling, then every round method on Adam, each row with the columns of the main table. A run not made yet shows as a dash.

```bash
python -m benchmark.rounds --data-root ~/Datasets                      # every dataset
python -m benchmark.rounds --data-root ~/Datasets --datasets covertype  # one
python -m benchmark.rounds --datasets covertype --tables-only           # the tables of what is recorded
```

The command runs every round and ceiling of the datasets asked for in one pool of processes, several per GPU (see below), skips runs already recorded, and prints the tables at the end. In the notebook, one cell after the initial-rate runs sweeps the whole study for `DATASET`, one run after another, and prints the same tables; the Figures section draws each round's figures into `images/<dataset>/rounds/<optimizer>_lr<rate>/`.

**Cost.** Going by the recorded runs, a round costs what the main table costs without Adam, SPS and Armijo: about 8 h of CIFAR-10 for five seeds, and some 57 h of run time over the five datasets, four of which shared the GPU three at a time. A round can take longer, since Adam's step is slower than SGD's and a method that stalls at its smallest candidate polls every other batch. SPS and Armijo add about 1.5 h of CIFAR-10 per ceiling, 15 h over the five datasets. These figures predate Efficient Relative Narrowing Polling, whose runs are not recorded yet; in a three-epoch CIFAR-10 smoke test it polled 42% of the batches, against 15% for Efficient Relative Polling.

## Running on a SLURM cluster

`python -m benchmark.rounds` above, and `python -m benchmark.pool` for any single sweep (it takes the flags of `python -m benchmark`), make the runs side by side, one process per `(method, seed)`, four per GPU unless `--workers` says otherwise. Each process is `python -m benchmark` for that method and seed, so the records are the ones a sequential sweep would write, except for `s_per_epoch`: runs that share a GPU slow each other down. Runs with a record are skipped, calibrated ablations wait for their own sweep's Efficient Polling run on the first seed, and each run logs to `logs/`, in a folder that mirrors the one its record goes to under `results/`.

`slurm/benchmark.sbatch` runs the whole study, every round and ceiling of every dataset, as one job. On the login node, clone the repository and create the environment the job activates:

```bash
git clone --branch dev https://github.com/luiz-linkezio/Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks.git
cd Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks
python3 -m venv .venv && .venv/bin/pip install -e ".[benchmark]"
mkdir -p logs
```

Any Python 3.10 or newer works, as long as the compute nodes see the interpreter it was created from. `uv` builds the same environment faster, but it locks its cache and the environment it installs into, and a file lock hangs for good on a home directory mounted over NFS with a broken lock daemon, which is what Apuana's did in September 2026. `pip` takes no such lock.

The job reads the datasets from `~/Datasets/`, in the folders the notebook's `DATA_DIRS` names (`DATA_ROOT` points it elsewhere). From a machine that has them:

```bash
rsync -av ~/Datasets/{cifar-10-python,cifar-100-python,MNIST,fashion-mnist,covertype} <user>@<login node>:Datasets/
```

Then submit it once, and follow it:

```bash
sbatch --nodelist=cluster-node7 slurm/benchmark.sbatch                 # every dataset
DATASETS="covertype cifar10" sbatch --nodelist=cluster-node7 slurm/benchmark.sbatch
squeue -u $USER
tail -f logs/polling-benchmark-<job id>.out
```

The job asks for two GPUs, 16 CPUs and 64 GB, the limits of the simple QoS of the cluster it was written for, Apuana at CIn/UFPE, and up to seven days on `long-simple`: the whole study takes about two days on two A100s, more than `short-simple` allows. The GPUs are untyped in SLURM there, so a node is picked with `--nodelist`, and the A100s of `long-simple` are `cluster-node7`'s. Options on the `sbatch` command line override the file's. A preempted job is requeued, and a job submitted again after a cancel carries on from the runs that had not finished. It ends by printing one table per initial rate of each dataset. The records are written to the clone on the cluster; bring them back with

```bash
rsync -av <user>@<login node>:Efficient-Polling-Based-Learning-Rate-Optimization-for-Neural-Networks/results/ results/
```

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
│   ├── efficient_relative_narrowing.py  # Efficient Relative Narrowing Polling
│   ├── baselines.py           # SPS and Armijo backtracking comparison optimizers
│   ├── _snapshot.py           # exact state save/restore for trial steps
│   ├── closures.py            # batch closures and selection criteria
│   └── training.py            # optional fit/train_epoch/evaluate helpers
├── benchmark/                 # the experiment, run by python -m benchmark and the notebook
│   ├── datasets.py            # the five dataset readers, no torchvision
│   ├── models.py              # SimpleCNN and SimpleMLP
│   ├── methods.py             # every configuration and its hyperparameters
│   ├── sweep.py               # runs configurations over seeds, records and reads back runs
│   ├── rounds.py              # the learning-rate study: rounds, ceilings, one table per initial rate
│   ├── pool.py                # runs side by side, several per GPU
│   └── plots.py               # the figures, drawn from the records
├── slurm/benchmark.sbatch     # the whole learning-rate study as one cluster job
├── notebooks/benchmark.ipynb  # drives the benchmark package, committed without outputs
├── tests/                     # pytest suite for the package and the benchmark
├── results/<dataset>/         # one JSON per (method, seed); rounds/ and ceilings/ hold the rounds
├── images/<dataset>/          # the figures of each dataset
├── docs/                      # how it works, results, reproducing; pt-br/ holds the Portuguese
├── models/                    # best checkpoints (.pt, gitignored)
├── logs/                      # logs of the pool and the cluster jobs (gitignored)
├── pyproject.toml
├── CHANGELOG.md
├── README.md
└── README(pt-br).md
```

To add a method, subclass `PollingOptimizer` in `src/`. It inherits exact snapshot and restore, the optimizer protocol and the `StepInfo` telemetry that `fit()` aggregates; set `optimizer_steps` to what the batch actually cost and the cost model stays honest. Then add an entry to `LABELS` and a branch to `build_optimizer()` in `benchmark/methods.py`, and the method joins the sweep, the tables and the figures.
