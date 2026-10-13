#!/bin/bash -l
# One-time setup on the cluster: installs and precompiles the Julia packages, then checks which Gurobi
# license works (without the cluster module, and with `module load gurobi/12.0.3`).
#
# Submit from the repository root:
#   mkdir -p logs
#   sbatch test/evaluation/slurm/setup.sbatch
# Result: logs/setup-<jobid>.out (look for the two "Gurobi OK:" lines at the end).

#SBATCH --partition=cpu-fast
#SBATCH --job-name=difog-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.out

module load "${JULIA_MODULE:-julia/1.12.1}"
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
unset LD_LIBRARY_PATH
export JULIA_NUM_PRECOMPILE_TASKS="$SLURM_CPUS_PER_TASK"

echo "== Installing and precompiling packages ($(date)) =="
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

check_gurobi() {
    julia --project=. -e 'using DiFoG; println("Gurobi OK: ", isassigned(DiFoG.GRB_ENV))'
}

echo "== Gurobi check WITHOUT the cluster module =="
check_gurobi || echo "Gurobi OK: false (julia failed)"

echo "== Gurobi check WITH module gurobi/12.0.3 =="
module load gurobi/12.0.3
check_gurobi || echo "Gurobi OK: false (julia failed)"

echo "== Done ($(date)) =="
