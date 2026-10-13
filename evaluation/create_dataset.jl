# evaluation/create_dataset.jl
#
# Creates (or completes) one seeded dataset: the graphs are drawn from `Xoshiro(seed)` by an ER generator and
# labelled by the chosen solver; only the labels are stored (see `create_dataset`). An interrupted run can
# simply be restarted, it continues where it stopped.
#
# Usage:
#   julia evaluation/create_dataset.jl <problem> <min_n> <max_n> <p> <seed> <num_instances> \
#         <solver> <time_limit_s> [result_root]
#   problem ∈ {misp, maxcut}
#   solver  ∈ {lns, lnsgreedy, gurobi}: lns = LNS-rand of Bresich et al. (d = 5, r = 2000 MIS / 1000 MaxCut), the
#             time limit is the LNS run time per instance (paper: 3 s); lnsgreedy = the same LNS started from the
#             degree greedy solution instead of a random one; gurobi = ILP, the time limit is Gurobi's.
#             LNS options: "lns:r=1:it=50000" sets r (random repairs per iteration) and an iteration limit; with
#             time_limit_s = 0 only the iteration limit applies (hardware-independent labels). Settings that differ
#             from the paper are part of the file name, e.g. <seed>_lns_r1_it50000.csv.
#   Gurobi threads: $GRB_THREADS if set, otherwise DiFoG's default (half of the CPU threads).
#
# Files: <result_root>/seeds/<seed>.csv (gurobi), <seed>_lns.csv (lns) and <seed>_lnsgreedy.csv (lnsgreedy).
# All describe the SAME graphs, so
# the two labellings of one seed can be created at the same time in two processes, e.g.
#   julia evaluation/create_dataset.jl maxcut 300 400 0.15 30040005 100 lns 3    &
#   julia evaluation/create_dataset.jl maxcut 300 400 0.15 30040005 100 gurobi 60 &
# Two processes must never work on the same file (same seed AND same solver).
#
# Note: an existing file is reused as is — the time limit is not part of the file name. Delete the file to
# regenerate it with a different time limit.

using Pkg
Pkg.activate(@__DIR__)

using DiFoG

const USAGE = "usage: create_dataset.jl <problem> <min_n> <max_n> <p> <seed> <num_instances> <solver> <time_limit_s> [result_root]"

"""
    build_solver(problem, solver_name, time_limit)

Constructs the labelling solver: the exact Gurobi solver of the problem, or the LNS wrapping it.
"""
function build_solver(problem::String, solver_spec::String, time_limit::Float64)
    # LNS options after the name: "lns:r=1:it=50000" (r = random repairs per iteration, it = iteration limit)
    parts = split(solver_spec, ':')
    solver_name = String(parts[1])
    opts = Dict{String,Int}()
    for o in parts[2:end]
        k, v = split(o, '=')
        k in ("r", "it") || error("unknown solver option '$k' (r | it)\n$USAGE")
        opts[String(k)] = parse(Int, v)
    end
    # the base solver only defines the problem for the LNS; its time limit is Gurobi's
    base_time = solver_name == "gurobi" || time_limit > 0 ? time_limit : 60.0
    base = problem == "misp"   ? MISPSolver(time_limit=base_time) :
           problem == "maxcut" ? MaxCutSolver(time_limit=base_time) :
           error("unknown problem '$problem' (misp | maxcut)\n$USAGE")
    if solver_name == "gurobi"
        isempty(opts) || error("solver options are only supported for the LNS\n$USAGE")
        return base
    end
    solver_name in ("lns", "lnsgreedy") || error("unknown solver '$solver_name' (lns | lnsgreedy | gurobi)\n$USAGE")
    return LNSSolver(base; ttime=time_limit, init=(solver_name == "lnsgreedy" ? :greedy : :random),
                     n_iter_repair=get(opts, "r", DiFoG.default_n_iter_repair(base)), titer=get(opts, "it", -1))
end

function main(args)
    length(args) in (8, 9) || error(USAGE)
    problem, solver_name = lowercase(args[1]), lowercase(args[7])
    min_n, max_n = parse(Int, args[2]), parse(Int, args[3])
    p = parse(Float64, args[4])
    seed, num_instances = parse(Int, args[5]), parse(Int, args[6])
    time_limit = parse(Float64, args[8])
    result_root = length(args) == 9 ? args[9] : "/home/e12023149/thesis/results/results_50_100/"

    haskey(ENV, "GRB_THREADS") && (DiFoG.GRB_THREADS[] = parse(Int, ENV["GRB_THREADS"]))

    generator = ERGraphGenerator(min_nodes=min_n, max_nodes=max_n, p=p)
    solver = build_solver(problem, solver_name, time_limit)

    @info "Creating dataset: $problem, $(generator), seed=$seed, $num_instances instances, solver=$(solver)" *
          (solver isa LNSSolver ? "" : ", Gurobi threads=$(DiFoG.GRB_THREADS[])")
    t = @elapsed path = create_dataset(seed, joinpath(result_root, "seeds"), generator, solver, num_instances)
    @info "Seeded dataset written to $path ($(round(t / 60, digits=1)) min)"
end

main(ARGS)
