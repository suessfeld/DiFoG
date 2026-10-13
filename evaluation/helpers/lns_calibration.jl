# test/helpers/lns_calibration.jl
#
# Calibrates an iteration limit for the LNS labels. For the first instances of a seeded dataset and both starts
# (random = LNS-rand, greedy = LNS-greedy) it reports
#   - the objective after 10², 10³, …, 10⁶ LNS iterations (the random decisions are seeded per instance, so a run
#     with k iterations is the prefix of every longer run: the columns form one consistent trajectory), and
#   - how many iterations this machine performs within a time limit (default 30 s) and the objective reached then.
# The iteration count equivalent to the time limit on the reference machine can then be used as a hardware-
# independent iteration limit (create_dataset.jl: "lns:r=1:it=<N>" with time limit 0).
# Run it on an otherwise idle machine: with other jobs on the same cores, fewer iterations fit into the time limit.
#
# Usage:
#   julia --project=. test/helpers/lns_calibration.jl <problem> <min_n> <max_n> <p> <seed> <n_inst> [r] [seconds]
#   problem ∈ {maxcut, misp}; r = random repairs per LNS iteration (default 1); seconds = time limit (default 30).
# Example (MaxCut ER-700-800, training seed):
#   julia --project=. test/helpers/lns_calibration.jl maxcut 700 800 0.15 70080014 5

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using Random
using Statistics
using Graphs
using GraphNeuralNetworks
using DiFoG

const CHECKPOINTS = [10^2, 10^3, 10^4, 10^5, 10^6]

function main(args)
    length(args) in 6:8 || error("expected 6 to 8 arguments; see the header of this file")
    problem = args[1]
    min_n, max_n = parse(Int, args[2]), parse(Int, args[3])
    p = parse(Float64, args[4])
    seed = parse(Int, args[5])
    n_inst = parse(Int, args[6])
    r = length(args) >= 7 ? parse(Int, args[7]) : 1
    seconds = length(args) >= 8 ? parse(Float64, args[8]) : 30.0

    base = problem == "maxcut" ? MaxCutSolver() : problem == "misp" ? MISPSolver() :
           error("Unknown problem '$problem' (expected maxcut or misp)")
    generator = ERGraphGenerator(min_n, max_n, p)
    rng = Random.Xoshiro(seed)
    objective(g, x) = DiFoG.evaluate(base, GNNGraph(g), vec(x) .> 0.5f0)

    # results[init] = per instance: (objectives at the checkpoints, objective and iterations within `seconds`)
    results = Dict(init => NamedTuple[] for init in (:random, :greedy))
    greedy_objs = Float64[]
    @printf("%s ER-%d-%d, seed %d, r = %d, time limit %.0f s, %d threads\n", problem, min_n, max_n, seed, r, seconds,
            Threads.nthreads())
    for idx in 1:n_inst
        g = generator(rng)
        greedy = DiFoG.evaluate(base, GNNGraph(g), DiFoG.greedy_mask(base, g))
        push!(greedy_objs, greedy)
        @printf("\ninstance %d: n = %d, m = %d, degree greedy = %.0f\n", idx, nv(g), ne(g), greedy)
        @printf("  %-7s | %s | %10s %12s\n", "start", join([@sprintf("%9s", "it=1e$(round(Int, log10(k)))") for k in CHECKPOINTS], " "),
                "$(round(Int, seconds)) s", "iterations")
        for init in (:random, :greedy)
            objs = Float64[]
            for k in CHECKPOINTS
                x, _, _ = DiFoG._run_lns(LNSSolver(base; ttime=-1, titer=k, n_iter_repair=r, init=init), g)
                push!(objs, objective(g, x))
            end
            t = @elapsed begin
                x, _, iters = DiFoG._run_lns(LNSSolver(base; ttime=seconds, n_iter_repair=r, init=init), g)
            end
            obj_t = objective(g, x)
            push!(results[init], (; objs, obj_t, iters, t))
            @printf("  %-7s | %s | %10.0f %12d  (%.0f it/s)\n", init, join([@sprintf("%9.0f", o) for o in objs], " "),
                    obj_t, iters, iters / t)
        end
    end

    # summary: objectives relative to the best solution found for each instance (over all runs), in %
    println("\n", "-"^100)
    @printf("mean objective relative to the best found per instance [%%] (%s, %d instances, r = %d):\n",
            problem, n_inst, r)
    best = [maximum(vcat(results[:random][i].objs, results[:greedy][i].objs,
                         results[:random][i].obj_t, results[:greedy][i].obj_t)) for i in 1:n_inst]
    @printf("  %-7s | %s | %10s %12s\n", "start", join([@sprintf("%9s", "it=1e$(round(Int, log10(k)))") for k in CHECKPOINTS], " "),
            "$(round(Int, seconds)) s", "mean it")
    for init in (:random, :greedy)
        rel = [mean(100 * results[init][i].objs[c] / best[i] for i in 1:n_inst) for c in eachindex(CHECKPOINTS)]
        rel_t = mean(100 * results[init][i].obj_t / best[i] for i in 1:n_inst)
        @printf("  %-7s | %s | %10.2f %12.0f\n", init, join([@sprintf("%9.2f", v) for v in rel], " "), rel_t,
                mean(r_.iters for r_ in results[init]))
    end
    @printf("  %-7s   degree greedy: %.2f\n", "", mean(100 * greedy_objs[i] / best[i] for i in 1:n_inst))
end

main(ARGS)
