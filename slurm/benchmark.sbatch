#!/bin/bash
#SBATCH --job-name=polling-benchmark
#SBATCH --partition=long-simple        # up to 7 days: the whole study takes about two, on two A100s
#SBATCH --gres=gpu:2                   # the GRES is untyped (gpu:N): choose A100 nodes with
                                       # --nodelist at submission, not here (on Apuana the
                                       # long-simple A100s are cluster-node7's)
#SBATCH --ntasks=1                     # one task: benchmark.pool starts the runs, not SLURM
#SBATCH --cpus-per-task=16             # the ceiling of the simple QoS; the pool shares it out
#SBATCH --mem=64G                      # every run holds its own copy of the dataset
#SBATCH --time=7-00:00:00
#SBATCH --requeue                      # a preempted job starts again from the runs left
#SBATCH --open-mode=append             # and a requeued job adds to its log instead of erasing it
#SBATCH --output=logs/%x-%j.out

# The whole learning-rate study in one job: every round (SGD and Adam, from 1,
# 1e-3 and 1e-7) and every SPS/Armijo ceiling, of every dataset, in one pool of
# processes over the job's GPUs (benchmark/rounds.py, benchmark/pool.py):
#
#     sbatch --nodelist=cluster-node7 slurm/benchmark.sbatch
#     DATASETS=covertype sbatch --nodelist=cluster-node7 slurm/benchmark.sbatch
#
# Each run is recorded in results/<dataset>/rounds/ or ceilings/ when it ends,
# and a run that has a record is skipped, so a job that was preempted, cancelled
# or out of time carries on from the runs left when it starts again. It ends by
# printing one table per initial LR of each dataset. Each run logs to
# logs/<dataset>/..., and this job to logs/polling-benchmark-<job>.out.
#
# Variables:
#   DATASETS   default "covertype cifar10 cifar100 mnist fashion_mnist"
#   DATA_ROOT  where the datasets are, laid out as in docs/reproducing.md (default ~/Datasets)
#   SEEDS      default "42 43 44 45 46"
#   WORKERS    runs at a time (default: 4 per GPU)
#   VENV       default .venv in the repository

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

NODE="${SLURMD_NODENAME:-$(hostname)}"
echo "job ${SLURM_JOB_ID:-?} | node $NODE | partition ${SLURM_JOB_PARTITION:-?} |" \
     "restarts ${SLURM_RESTART_COUNT:-0} | $(date '+%F %T')"

VENV="${VENV:-$SLURM_SUBMIT_DIR/.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: no environment at $VENV. Create it on the login node:"
    echo "           python3 -m venv .venv && .venv/bin/pip install -e '.[benchmark]'"
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

read -r -a DATASET_LIST <<< "${DATASETS:-covertype cifar10 cifar100 mnist fashion_mnist}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44 45 46}"

python -m benchmark.rounds \
    --data-root "${DATA_ROOT:-$HOME/Datasets}" \
    --datasets "${DATASET_LIST[@]}" \
    --seeds "${SEED_LIST[@]}" \
    ${WORKERS:+--workers "$WORKERS"}
