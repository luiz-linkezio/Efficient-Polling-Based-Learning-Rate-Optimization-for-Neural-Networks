#!/bin/bash
#SBATCH --job-name=polling-benchmark
#SBATCH --partition=short-simple       # up to 2 days; one round of the slowest dataset takes hours
#SBATCH --gres=gpu:2                   # the GRES is untyped (gpu:N): choose A100 nodes with
                                       # --nodelist at submission, not here
#SBATCH --ntasks=1                     # one task: benchmark.pool starts the runs, not SLURM
#SBATCH --cpus-per-task=16             # the ceiling of the simple QoS; the pool shares it out
#SBATCH --mem=64G                      # every run holds its own copy of the dataset
#SBATCH --time=1-00:00:00
#SBATCH --requeue                      # a preempted job starts again from the runs left
#SBATCH --open-mode=append             # and a requeued job adds to its log instead of erasing it
#SBATCH --output=logs/%x-%j.out

# One learning-rate round, or one SPS/Armijo ceiling, of one dataset:
#
#     DATASET=covertype ROUND=sgd:1 sbatch slurm/benchmark.sbatch
#     DATASET=cifar10 CEILING=1e-7 sbatch --nodelist=cluster-node9 slurm/benchmark.sbatch
#
# The runs go side by side, several per GPU (benchmark/pool.py), each recorded
# in results/<dataset>/rounds/ or ceilings/ when it ends. A run that already has
# a record is skipped, so a job that was preempted, cancelled or out of time is
# resumed by submitting it again. Each run logs to logs/<dataset>/..., and this
# job to logs/polling-benchmark-<job>.out.
#
# Variables:
#   DATASET    cifar10 | cifar100 | mnist | fashion_mnist | covertype (required)
#   ROUND      OPTIMIZER:RATE, one of sgd|adam : 1|1e-3|1e-7
#   CEILING    1 | 1e-3 | 1e-7; ROUND or CEILING, not both
#   DATA_ROOT  where the datasets are, laid out as in docs/reproducing.md (default ~/Datasets)
#   DATA_DIR   the dataset's own folder, when it is not where DATA_ROOT puts it
#   SEEDS      default "42 43 44 45 46"
#   WORKERS    runs at a time (default: 4 per GPU)
#   VENV       default .venv in the repository

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

NODE="${SLURMD_NODENAME:-$(hostname)}"
echo "job ${SLURM_JOB_ID:-?} | node $NODE | partition ${SLURM_JOB_PARTITION:-?} |" \
     "restarts ${SLURM_RESTART_COUNT:-0} | $(date '+%F %T')"

if [[ -z "${DATASET:-}" ]]; then
    echo "ERROR: set DATASET, such as DATASET=covertype ROUND=sgd:1 sbatch slurm/benchmark.sbatch"
    exit 1
fi
if [[ -n "${ROUND:-}" && -n "${CEILING:-}" ]]; then
    echo "ERROR: ROUND=$ROUND and CEILING=$CEILING: a job runs one or the other"
    exit 1
elif [[ -n "${ROUND:-}" ]]; then
    VARIANT=(--round "$ROUND")
elif [[ -n "${CEILING:-}" ]]; then
    VARIANT=(--ceiling "$CEILING")
else
    echo "ERROR: set ROUND (such as ROUND=adam:1e-3) or CEILING (such as CEILING=1e-7)"
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-$HOME/Datasets}"
case "$DATASET" in
    cifar10)       FOLDER=cifar-10-python/cifar-10-batches-py ;;
    cifar100)      FOLDER=cifar-100-python ;;
    mnist)         FOLDER=MNIST ;;
    fashion_mnist) FOLDER=fashion-mnist ;;
    covertype)     FOLDER=covertype ;;
    *) echo "ERROR: unknown DATASET=$DATASET"; exit 1 ;;
esac
DATA_DIR="${DATA_DIR:-$DATA_ROOT/$FOLDER}"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: no dataset at $DATA_DIR. Copy it from a machine that has it:"
    echo "           rsync -av ~/Datasets/$FOLDER/ <user>@<login node>:Datasets/$FOLDER/"
    exit 1
fi

VENV="${VENV:-$SLURM_SUBMIT_DIR/.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: no environment at $VENV. Create it on the login node:"
    echo "           uv venv --python 3.12 && uv pip install -e '.[benchmark]'"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

nvidia-smi -L || echo "WARNING: nvidia-smi did not answer; the job may have no GPU."
# nvidia-smi seeing a GPU does not mean torch can use it, and a sweep that falls
# back to the CPU still runs, a hundred times slower. Failing here takes seconds.
if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: torch sees no GPU. Check the torch build against the node's driver:"
    echo "           python -c 'import torch; print(torch.__version__, torch.version.cuda)'"
    exit 1
fi

read -r -a SEED_LIST <<< "${SEEDS:-42 43 44 45 46}"

python -m benchmark.pool \
    ${WORKERS:+--workers "$WORKERS"} \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --seeds "${SEED_LIST[@]}" \
    "${VARIANT[@]}"
