# evaluation/helpers/gurobi_convergence.jl
#
# Gurobi convergence study: solves dataset instances with a long time limit and records the incumbent
# and the dual bound over time, to determine after how many seconds Gurobi stops improving (and thus a
# sensible time limit for generating the reference labels).
#
# The instances are exactly those of the seeded dataset: the k-th instance is the k-th graph drawn from
# `Xoshiro(seed)` by the same generator, as in `create_dataset`. The ILP is the same one used for the labels.
#
# Usage:
#   julia evaluation/helpers/gurobi_convergence.jl <problem> <min_n> <max_n> <p> <seed> \
#         <first_idx> <last_idx> <time_limit_s> <outdir>
#   problem ∈ {maxcut, misp}; Gurobi threads = $SLURM_CPUS_PER_TASK ÷ 2 (one per physical core), at least 1,
#   overridable with $GRB_THREADS.
#
# Output: one CSV per instance, <outdir>/traj_<problem>_<seed>_<idx>.csv, with rows
#   kind,time,incumbent,bound     kind ∈ {sol, bound, final}
# "sol" rows are improving solutions, "bound" rows are dual-bound improvements (integer steps), and the
# single "final" row holds the solve time, final objective and final bound. Analyze the directory with
# evaluation/helpers/analyze_gurobi_convergence.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Graphs
using JuMP
using Gurobi
using DiFoG

"""
    make_recorder(rows)

Returns a Gurobi callback that appends `(kind, time, incumbent, bound)` to `rows` whenever the incumbent
improves or the dual bound improves by a full integer step (all considered objectives are integral).
"""
function make_recorder(rows::Vector{Tuple{String,Float64,Float64,Float64}})
    best  = Ref(-Inf)
    bound = Ref(Inf)
    return function (cb_data, cb_where::Cint)
        t = Ref{Cdouble}()
        if cb_where == Gurobi.GRB_CB_MIPSOL
            obj = Ref{Cdouble}(); bnd = Ref{Cdouble}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_RUNTIME, t)
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJ, obj)
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJBND, bnd)
            if obj[] > best[] + 1e-9
                best[]  = obj[]
                bound[] = min(bound[], bnd[])
                push!(rows, ("sol", t[], best[], bound[]))
            end
        elseif cb_where == Gurobi.GRB_CB_MIP
            bnd = Ref{Cdouble}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIP_OBJBND, bnd)
            if floor(bnd[] + 1e-6) < floor(bound[] + 1e-6)
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_RUNTIME, t)
                bound[] = bnd[]
                push!(rows, ("bound", t[], best[], bound[]))
            end
        end
        return
    end
end

"""
    dataset_graph(generator, seed, idx)

Returns the `idx`-th (1-based) graph of the dataset with this seed, replaying the generator's RNG stream.
"""
function dataset_graph(generator::GraphGenerator, seed::Integer, idx::Integer)
    rng = Random.Xoshiro(seed)
    g = generator(rng)
    for _ in 2:idx
        g = generator(rng)
    end
    return g
end

function solve_with_trajectory(problem::String, g::SimpleGraph, time_limit::Float64, threads::Int)
    nsp_problem = problem == "maxcut" ? MAXCUT() : problem == "misp" ? MIS(0) :
                  error("Unknown problem '$problem' (expected maxcut or misp)")
    inst = NSPInstance(nsp_problem, g, nv(g), ne(g), Ref{Union{Nothing,JuMP.Model}}(nothing), (;))

    rows = Tuple{String,Float64,Float64,Float64}[]
    sol, status, rel_gap, solve_time = ILP!(inst; silent=true, time_limit, warn_time_limit=false,
                                            env=DiFoG.GRB_ENV[], threads, callback=make_recorder(rows))
    model = inst.ilp_model[]
    push!(rows, ("final", solve_time, objective_value(model), objective_bound(model)))
    return rows, status, rel_gap
end

function main(args)
    length(args) == 9 || error("expected 9 arguments, got $(length(args)); see the header of this file")
    problem = args[1]
    min_n, max_n = parse(Int, args[2]), parse(Int, args[3])
    p = parse(Float64, args[4])
    seed = parse(Int, args[5])
    first_idx, last_idx = parse(Int, args[6]), parse(Int, args[7])
    time_limit = parse(Float64, args[8])
    outdir = args[9]

    cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "2"))
    threads = parse(Int, get(ENV, "GRB_THREADS", string(max(1, cpus ÷ 2))))
    isassigned(DiFoG.GRB_ENV) || error("No Gurobi environment — check the license (module load gurobi / GRB_LICENSE_FILE).")

    mkpath(outdir)
    generator = ERGraphGenerator(min_n, max_n, p)
    for idx in first_idx:last_idx
        g = dataset_graph(generator, seed, idx)
        @info "Instance $idx of seed $seed: n=$(nv(g)), m=$(ne(g)), time limit $(time_limit)s, $threads thread(s)"
        rows, status, rel_gap = solve_with_trajectory(problem, g, time_limit, threads)

        path = joinpath(outdir, "traj_$(problem)_$(seed)_$(idx).csv")
        open(path, "w") do io
            println(io, "kind,time,incumbent,bound")
            for (kind, t, inc, bnd) in rows
                println(io, kind, ',', t, ',', inc, ',', bnd)
            end
        end
        final = rows[end]
        @info "Instance $idx done: status=$status, obj=$(final[3]), bound=$(final[4]), gap=$(rel_gap), time=$(round(final[2], digits=1))s → $path"
    end
end

main(ARGS)
