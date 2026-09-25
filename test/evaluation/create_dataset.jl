# test/evaluation/create_dataset.jl
#
# Creates (or completes) one seeded dataset: the graphs are drawn from `Xoshiro(seed)` by an ER generator and
# labelled by the chosen solver; only the labels are stored (see `create_dataset`). An interrupted run can
# simply be restarted, it continues where it stopped.
#
# Usage:
#   julia --project=. test/evaluation/create_dataset.jl <problem> <min_n> <max_n> <p> <seed> <num_instances> \
#         <solver> <time_limit_s> [result_root]
#   problem ∈ {misp, maxcut}
#   solver  ∈ {lns, gurobi}: lns = LNS-rand of Bresich et al. (d = 5, r = 2000 MIS / 1000 MaxCut), the time limit
#             is the LNS run time per instance (paper: 3 s); gurobi = ILP, the time limit is Gurobi's.
#   Gurobi threads: $GRB_THREADS if set, otherwise DiFoG's default (half of the CPU threads).
#
# Files: <result_root>/seeds/<seed>.csv (gurobi) and <seed>_lns.csv (lns). Both describe the SAME graphs, so
# the two labellings of one seed can be created at the same time in two processes, e.g.
#   julia --project=. test/evaluation/create_dataset.jl maxcut 300 400 0.15 30040005 100 lns 3    &
#   julia --project=. test/evaluation/create_dataset.jl maxcut 300 400 0.15 30040005 100 gurobi 60 &
# Two processes must never work on the same file (same seed AND same solver).
#
# Note: an existing file is reused as is — the time limit is not part of the file name. Delete the file to
# regenerate it with a different time limit.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DiFoG

const USAGE = "usage: create_dataset.jl <problem> <min_n> <max_n> <p> <seed> <num_instances> <solver> <time_limit_s> [result_root]"

"""
    build_solver(problem, solver_name, time_limit)

Constructs the labelling solver: the exact Gurobi solver of the problem, or the LNS wrapping it.
"""
function build_solver(problem::String, solver_name::String, time_limit::Float64)
    base = problem == "misp"   ? MISPSolver(time_limit=time_limit) :
           problem == "maxcut" ? MaxCutSolver(time_limit=time_limit) :
           error("unknown problem '$problem' (misp | maxcut)\n$USAGE")
    solver_name == "gurobi" && return base
    solver_name == "lns" && return LNSSolver(base; ttime=time_limit)
    error("unknown solver '$solver_name' (lns | gurobi)\n$USAGE")
end

function main(args)
    length(args) in (8, 9) || error(USAGE)
    problem, solver_name = lowercase(args[1]), lowercase(args[7])
    min_n, max_n = parse(Int, args[2]), parse(Int, args[3])
    p = parse(Float64, args[4])
    seed, num_instances = parse(Int, args[5]), parse(Int, args[6])
    time_limit = parse(Float64, args[8])
    result_root = length(args) == 9 ? args[9] : "/home/e12023149/thesis/results"

    haskey(ENV, "GRB_THREADS") && (DiFoG.GRB_THREADS[] = parse(Int, ENV["GRB_THREADS"]))

    generator = ERGraphGenerator(min_nodes=min_n, max_nodes=max_n, p=p)
    solver = build_solver(problem, solver_name, time_limit)

    @info "Creating dataset: $problem, $(generator), seed=$seed, $num_instances instances, solver=$(solver)" *
          (solver isa LNSSolver ? "" : ", Gurobi threads=$(DiFoG.GRB_THREADS[])")
    t = @elapsed path = create_dataset(seed, joinpath(result_root, "seeds"), generator, solver, num_instances)
    @info "Seeded dataset written to $path ($(round(t / 60, digits=1)) min)"
end

main(ARGS)
