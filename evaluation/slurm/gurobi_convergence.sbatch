#!/bin/bash -l
# Gurobi convergence study as a Slurm array: one dataset instance per array task.
#
# Submit from the repository root, e.g. (defaults = MaxCut on ER-300, p = 0.1, training seed):
#   mkdir -p logs
#   sbatch evaluation/slurm/gurobi_convergence.sbatch
#   sbatch --export=ALL,PROBLEM=misp,NMIN=300,NMAX=400,P=0.15,SEED=30040001 evaluation/slurm/gurobi_convergence.sbatch
# If the setup job showed that Gurobi needs the cluster module, add GUROBI_MODULE=gurobi/12.0.3 to --export.
# Analyze afterwards (cheap), e.g. as a small batch job:
#   sbatch --partition=cpu-fast --mem=2G --time=00:05:00 --output=logs/analyze-%j.out --wrap="module load julia/1.12.1; julia evaluation/helpers/analyze_gurobi_convergence.jl results/gurobi_convergence/<problem>_<nmin>_<nmax>_<p>_<seed>"
#
# Two Slurm CPUs = one physical core on the SMT nodes, so single-threaded Gurobi does not share its core
# with another job (otherwise the measured times are distorted). --time = TLIMIT + startup margin.

#SBATCH --partition=cpu-fast
#SBATCH --job-name=grb-conv
#SBATCH --array=1-50%50
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/grb-conv-%A_%a.out

module load "${JULIA_MODULE:-julia/1.12.1}"
if [ -n "${GUROBI_MODULE:-}" ]; then module load "$GUROBI_MODULE"; fi
set -eo pipefail

PROBLEM=${PROBLEM:-maxcut}
NMIN=${NMIN:-300}
NMAX=${NMAX:-300}
P=${P:-0.1}
SEED=${SEED:-30300001}
TLIMIT=${TLIMIT:-600}
JULIA=${JULIA:-julia}
OUTDIR=${OUTDIR:-results/gurobi_convergence/${PROBLEM}_${NMIN}_${NMAX}_${P}_${SEED}}

cd "$SLURM_SUBMIT_DIR"
unset LD_LIBRARY_PATH   # avoid picking up system CUDA libraries when DiFoG loads CUDA.jl

echo "Task $SLURM_ARRAY_TASK_ID on $(hostname): $PROBLEM n=[$NMIN,$NMAX] p=$P seed=$SEED limit=${TLIMIT}s"
"$JULIA" evaluation/helpers/gurobi_convergence.jl \
    "$PROBLEM" "$NMIN" "$NMAX" "$P" "$SEED" "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_ID" "$TLIMIT" "$OUTDIR"
