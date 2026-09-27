# test/helpers/lns_init_comparison.jl
#
# Compares the degree greedy, LNS-rand (random start) and the LNS started from the greedy solution on the first
# instances of a seeded dataset, to check whether the LNS reference is better than the greedy baseline.
# The k-th instance is the k-th graph drawn from `Xoshiro(seed)` by the generator, as in `create_dataset`.
#
# Usage:
#   julia --project=. test/helpers/lns_init_comparison.jl <problem> <min_n> <max_n> <p> <seed> <n_inst> [ttime]
#   problem ∈ {maxcut, misp}; ttime = LNS time limit in seconds (default 3).
# Example (MaxCut ER-700-800, training seed):
#   julia --project=. test/helpers/lns_init_comparison.jl maxcut 700 800 0.15 70080004 5
#
# Results (Slurm cluster, ER-700-800, p = 0.15, first 5 instances of the training datasets; objective values).
# They motivated `init = :greedy` for the ER-700-800 labels: within 3 s, LNS-rand stays below the degree greedy on
# MaxCut, while the LNS started from the greedy solution is above it on both problems.
#
#   MISP, seed 70080001, 3 s:
#    idx     n       m |     greedy   LNS-rand LNS-greedy |   rand/gr  lnsgr/gr
#      1   729   39854 |         33         39         39 |   118.18%   118.18%
#      2   754   42571 |         33         37         39 |   112.12%   118.18%
#      3   705   37492 |         37         38         40 |   102.70%   108.11%
#      4   702   37106 |         32         37         39 |   115.62%   121.88%
#      5   787   46508 |         34         36         39 |   105.88%   114.71%
#   mean relative to the greedy: LNS-rand 110.90%, LNS-greedy 116.21%
#
#   MaxCut, seed 70080004, 3 s:
#    idx     n       m |     greedy   LNS-rand LNS-greedy |   rand/gr  lnsgr/gr
#      1   773   44738 |      24591      23731      24749 |    96.50%   100.64%
#      2   714   38315 |      20942      20536      21095 |    98.06%   100.73%
#      3   764   44183 |      24192      23541      24282 |    97.31%   100.37%
#      4   706   37242 |      20344      20017      20468 |    98.39%   100.61%
#      5   777   45553 |      24823      24113      24956 |    97.14%   100.54%
#   mean relative to the greedy: LNS-rand 97.48%, LNS-greedy 100.58%
#
#   MaxCut, seed 70080004, 10 s:
#    idx     n       m |     greedy   LNS-rand LNS-greedy |   rand/gr  lnsgr/gr
#      1   773   44738 |      24591      24327      24846 |    98.93%   101.04%
#      2   714   38315 |      20942      21010      21255 |   100.32%   101.49%
#      3   764   44183 |      24192      24060      24341 |    99.45%   100.62%
#      4   706   37242 |      20344      20440      20681 |   100.47%   101.66%
#      5   777   45553 |      24823      24816      25115 |    99.97%   101.18%
#   mean relative to the greedy: LNS-rand 99.83%, LNS-greedy 101.20%

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
