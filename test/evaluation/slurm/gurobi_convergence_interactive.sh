#!/bin/bash -l
# Gurobi convergence check without sbatch: runs gurobi_convergence.jl for both instance sizes (ER-300-400,
# then ER-700-800) and both problems one after the other in the current allocation, and prints the analysis
# (how far the incumbent at 1/3/5/10/30/60/120/300 s is from the final one) after each configuration.
#
# Start it interactively from the repository root, e.g.
#   mkdir -p logs
#   srun --partition=cpu-fast --cpus-per-task=4 --mem=16G --time=04:00:00 --pty \
#        bash test/evaluation/slurm/gurobi_convergence_interactive.sh 2>&1 | tee logs/grb-conv-interactive.log
#
# Settings (environment variables, defaults in brackets):
#   TLIMIT    Gurobi time limit per instance in seconds [300]; must exceed the time limit you want to judge
#   N_INST    instances per configuration [5]
#   PROBLEMS  problems to check ["maxcut misp"]
#   SIZES     node ranges as min:max ["300:400 700:800"]
#   P         edge probability [0.15]
#   GRB_THREADS  Gurobi threads [cpus-per-task / 2]; use the value you will use for the dataset generation,
#                since the time needed depends on it
#   GUROBI_MODULE  e.g. gurobi/12.0.3, if the license needs the cluster module (see setup.sbatch)
# Runtime ≈ |PROBLEMS| × |SIZES| × N_INST × TLIMIT, i.e. 4 × 5 × 300 s ≈ 100 min with the defaults.
#
# The instances are the first N_INST graphs of the TRAINING datasets (seeds below), i.e. exactly the graphs the
# labels will be created for. Trajectories: results/gurobi_convergence/<problem>_<min>_<max>_<p>_<seed>/.

# The julia module sets JULIA_DEPOT_PATH itself; keep a depot the caller exported before (e.g. a
# cluster-specific one, JULIA_DEPOT_PATH="$HOME/.julia_cluster:") instead of the module's default.
CALLER_DEPOT="${JULIA_DEPOT_PATH:-}"
module load "${JULIA_MODULE:-julia/1.12.1}"
if [ -n "${GUROBI_MODULE:-}" ]; then module load "$GUROBI_MODULE"; fi
if [ -n "$CALLER_DEPOT" ]; then export JULIA_DEPOT_PATH="$CALLER_DEPOT"; fi
echo "Julia depot: ${JULIA_DEPOT_PATH:-<default ~/.julia>}"
set -eo pipefail
unset LD_LIBRARY_PATH   # avoid picking up system CUDA libraries when DiFoG loads CUDA.jl

TLIMIT=${TLIMIT:-300}
N_INST=${N_INST:-5}
PROBLEMS=${PROBLEMS:-"maxcut misp"}
SIZES=${SIZES:-"300:400 700:800"}
P=${P:-0.15}
JULIA=${JULIA:-julia}

# training seeds of the datasets (evaluate.jl): 3004000x for ER-300-400, 7008000x for ER-700-800
seed_for() {  # <problem> <min_n>
    case "$1:$2" in
        misp:300)   echo 30040001 ;;
        maxcut:300) echo 30040004 ;;
        misp:700)   echo 70080001 ;;
        maxcut:700) echo 70080004 ;;
        *)          echo 99990001 ;;   # any other size: an arbitrary fixed seed
    esac
}

# repository root = three levels above this script, independent of where srun/bash was started
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
for size in $SIZES; do
    NMIN=${size%%:*}
    NMAX=${size##*:}
    for PROBLEM in $PROBLEMS; do
        SEED=$(seed_for "$PROBLEM" "$NMIN")
        OUTDIR=results/gurobi_convergence/${PROBLEM}_${NMIN}_${NMAX}_${P}_${SEED}
        echo "=================================================================================="
        echo "== $PROBLEM on ER-$NMIN-$NMAX (p=$P), seed $SEED, instances 1..$N_INST, ${TLIMIT}s each  ($(date))"
        echo "=================================================================================="
        "$JULIA" --project=. test/evaluation/gurobi_convergence.jl \
            "$PROBLEM" "$NMIN" "$NMAX" "$P" "$SEED" 1 "$N_INST" "$TLIMIT" "$OUTDIR"
        "$JULIA" test/evaluation/analyze_gurobi_convergence.jl "$OUTDIR"
    done
done
echo "== Done ($(date))"
