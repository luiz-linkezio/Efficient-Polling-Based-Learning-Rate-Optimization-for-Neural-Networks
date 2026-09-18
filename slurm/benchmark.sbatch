#!/bin/bash
#SBATCH --job-name=polling-benchmark
#SBATCH --partition=short-simple       # up to 2 days; the job requeues itself before that, below
#SBATCH --gres=gpu:2                   # the GRES is untyped (gpu:N): choose A100 nodes with
                                       # --nodelist at submission, not here
#SBATCH --ntasks=1                     # one task: benchmark.pool starts the runs, not SLURM
#SBATCH --cpus-per-task=16             # the ceiling of the simple QoS; the pool shares it out
#SBATCH --mem=64G                      # every run holds its own copy of the dataset
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@900            # 15 min before the limit, tell this script so
#SBATCH --requeue                      # a preempted or timed-out job starts again from the runs left
#SBATCH --open-mode=append             # and a requeued job adds to its log instead of erasing it
#SBATCH --output=logs/%x-%j.out

# The whole learning-rate study in one job: every round (SGD and Adam, from 1,
# 1e-3 and 1e-7) and every SPS/Armijo ceiling, of every dataset, in one pool of
# processes over the job's GPUs (benchmark/rounds.py, benchmark/pool.py):
#
#     sbatch --nodelist=cluster-node9 slurm/benchmark.sbatch
#     DATASETS=covertype sbatch --nodelist=cluster-node9 slurm/benchmark.sbatch
#
# Each run is recorded in results/<dataset>/rounds/ or ceilings/ when it ends,
# and a run that has a record is skipped. So the job does not need to fit in the
# partition's two days: 15 minutes before the limit it puts itself back in the
# queue, and the next start carries on from the runs left. It ends by printing
# one table per initial LR of each dataset. Each run logs to logs/<dataset>/...,
# and this job to logs/polling-benchmark-<job>.out.
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

# Near the time limit SLURM sends USR1 to this script. Requeueing stops every
# process of the job (the finished runs are already on disk) and queues it again.
requeue() {
    echo "time limit near: requeueing job $SLURM_JOB_ID | $(date '+%F %T')"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1

# In the background, so the trap can run while the study does: bash runs a trap
# only between commands, and `wait` is one it can interrupt.
python -m benchmark.rounds \
    --data-root "${DATA_ROOT:-$HOME/Datasets}" \
    --datasets "${DATASET_LIST[@]}" \
    --seeds "${SEED_LIST[@]}" \
    ${WORKERS:+--workers "$WORKERS"} &
STUDY=$!
status=0
while true; do
    wait "$STUDY" && status=0 || status=$?
    # A trapped signal ends `wait` early with a code above 128 while the study
    # still runs; any other return is the study's own exit.
    kill -0 "$STUDY" 2>/dev/null || break
done
exit "$status"
