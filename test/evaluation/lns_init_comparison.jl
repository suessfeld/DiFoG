# test/evaluation/lns_init_comparison.jl
#
# Compares the degree greedy, LNS-rand (random start) and the LNS started from the greedy solution on the first
# instances of a seeded dataset, to check whether the LNS reference is better than the greedy baseline.
# The k-th instance is the k-th graph drawn from `Xoshiro(seed)` by the generator, as in `create_dataset`.
#
# Usage:
#   julia --project=. test/evaluation/lns_init_comparison.jl <problem> <min_n> <max_n> <p> <seed> <n_inst> [ttime]
#   problem ∈ {maxcut, misp}; ttime = LNS time limit in seconds (default 3).
# Example (MaxCut ER-700-800, training seed):
#   julia --project=. test/evaluation/lns_init_comparison.jl maxcut 700 800 0.15 70080004 5

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using Random
using Statistics
using Graphs
using GraphNeuralNetworks
using DiFoG

function main(args)
    length(args) in (6, 7) || error("expected 6 or 7 arguments; see the header of this file")
    problem = args[1]
    min_n, max_n = parse(Int, args[2]), parse(Int, args[3])
    p = parse(Float64, args[4])
    seed = parse(Int, args[5])
    n_inst = parse(Int, args[6])
    ttime = length(args) == 7 ? parse(Float64, args[7]) : 3.0

    base = problem == "maxcut" ? MaxCutSolver() : problem == "misp" ? MISPSolver() :
           error("Unknown problem '$problem' (expected maxcut or misp)")
    lns_rand = LNSSolver(base; ttime, init=:random)
    lns_greedy = LNSSolver(base; ttime, init=:greedy)
    generator = ERGraphGenerator(min_n, max_n, p)
    rng = Random.Xoshiro(seed)

    objective(g, x) = DiFoG.evaluate(base, GNNGraph(g), vec(x) .> 0.5f0)
    rows = NamedTuple[]
    @printf("%4s %5s %7s | %10s %10s %10s | %9s %9s\n", "idx", "n", "m", "greedy", "LNS-rand", "LNS-greedy",
            "rand/gr", "lnsgr/gr")
    for idx in 1:n_inst
        g = generator(rng)
        greedy = DiFoG.evaluate(base, GNNGraph(g), DiFoG.greedy_mask(base, g))
        rand_obj = objective(g, first(lns_rand(g)))
        greedy_obj = objective(g, first(lns_greedy(g)))
        push!(rows, (; greedy, rand_obj, greedy_obj))
        @printf("%4d %5d %7d | %10.0f %10.0f %10.0f | %8.2f%% %8.2f%%\n", idx, nv(g), ne(g), greedy, rand_obj,
                greedy_obj, 100 * rand_obj / greedy, 100 * greedy_obj / greedy)
    end
    mean_ratio(f) = mean(100 * f(r) / r.greedy for r in rows)
    println("-"^78)
    @printf("mean objective relative to the greedy (%s, %d instances, %.1f s): LNS-rand %.2f%%, LNS-greedy %.2f%%\n",
            problem, n_inst, ttime, mean_ratio(r -> r.rand_obj), mean_ratio(r -> r.greedy_obj))
end

main(ARGS)
