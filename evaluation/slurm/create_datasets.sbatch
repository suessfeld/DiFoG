#!/bin/bash -l
# Creates label files as a Slurm job array: one array task per line of a job file (one create_dataset.jl call),
# with one CPU core each, so all files are labelled in parallel. Intended for the LNS labels
# (label_jobs_lns.txt): they are limited by an iteration count, not by time, so the labels are the same on every
# node, however fast it is. The Gurobi labels (label_jobs_gurobi.txt) are time-limited and should be created on
# one known machine instead (conan), since their quality depends on the speed of the node.
#
# Submit from the repository root, with the cluster-specific Julia depot exported (it is passed on to the tasks):
#   export JULIA_DEPOT_PATH=$HOME/.julia_cluster:
#   mkdir -p logs/labels
#   JOBS=evaluation/slurm/label_jobs_lns.txt
#   sbatch --array=1-$(grep -c . $JOBS) --export=ALL,JOBS=$JOBS evaluation/slurm/create_datasets.sbatch
# Precompile once before (julia --project=evaluation -e 'using DiFoG'), so the tasks do not all compile at the same time.
# Options: RESULT_ROOT (labels go to $RESULT_ROOT/seeds; default $HOME/difog/results), a limit of simultaneous
# tasks (--array=1-18%10), another partition (--partition=...; the longest file needs about 1.5 days).
# An interrupted task (e.g. at the time limit) can simply be submitted again: create_dataset.jl continues each file
# where it stopped. Two tasks must never work on the same file (same seed and solver).

#SBATCH --partition=cpu-fast
#SBATCH --job-name=difog-labels
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/labels/%x-%A_%a.out

# The julia module sets JULIA_DEPOT_PATH itself; keep the depot exported by the caller.
CALLER_DEPOT="${JULIA_DEPOT_PATH:-}"
module load "${JULIA_MODULE:-julia/1.12.1}"
module load "${GUROBI_MODULE:-gurobi/12.0.3}"
if [ -n "$CALLER_DEPOT" ]; then export JULIA_DEPOT_PATH="$CALLER_DEPOT"; fi
set -eo pipefail
unset LD_LIBRARY_PATH   # avoid picking up system CUDA libraries when DiFoG loads CUDA.jl

cd "$SLURM_SUBMIT_DIR"
export GRB_THREADS=1 JULIA_NUM_THREADS=1
export RESULT_ROOT="${RESULT_ROOT:-$HOME/difog/results}"

: "${JOBS:?set JOBS to the job file (see the header of this script)}"
line=$(grep . "$JOBS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$line" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $JOBS"; exit 1; }

echo "== task $SLURM_ARRAY_TASK_ID on $(hostname), depot ${JULIA_DEPOT_PATH:-<default>} ($(date))"
echo "== $line"
eval "$line"
echo "== done ($(date))"
