# test/helpers/analyze_gurobi_convergence.jl
#
# Summarizes the trajectories written by gurobi_convergence.jl: for each checkpoint time T it reports how
# far the incumbent at T is from the incumbent at the end of the run, and how many instances have already
# reached their final incumbent. Also reports when the last improvement happened and the remaining gap to
# the dual bound. Only needs the Julia standard library.
#
# Usage:
#   julia test/helpers/analyze_gurobi_convergence.jl <dir> [checkpoints...]
#   default checkpoints: 1 3 5 10 30 60 120 300 600
#
# Results (Slurm cluster, 26.09.2026; gurobi_convergence.jl with a 300 s limit and 1 thread, p = 0.15, first 5
# instances of the training datasets). "short" = how far the incumbent at T is below the final incumbent.
# Main finding: on MaxCut, Gurobi finds its final incumbent within seconds and then stalls (weak LP relaxation,
# gap to the dual bound stays at 43 % / 78 %), so more time does not help; on MISP it keeps improving for tens of
# seconds (at 3 s still 5.8 % / 26 % short, at 60 s within 0.6 % / 1.4 %). No instance is proven optimal.
#
#   MaxCut ER-300-400, seed 30040004
#      T [s] | mean short [%] |  max short [%] |  reached final [%] |   no sol [%]
#        1.0 |         0.2250 |         0.4890 |                0.0 |          0.0
#        3.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#        5.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#       10.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#       30.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#       60.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#      120.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#      300.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#   last improvement [s]: median 1.2 | 90% 1.4 | max 1.5; remaining gap to the bound: mean 43.37 % | max 49.15 %
#
#   MISP ER-300-400, seed 30040001
#      T [s] | mean short [%] |  max short [%] |  reached final [%] |   no sol [%]
#        1.0 |        16.2562 |        28.5714 |                0.0 |          0.0
#        3.0 |         5.7825 |        11.4286 |                0.0 |          0.0
#        5.0 |         2.9254 |         3.0303 |                0.0 |          0.0
#       10.0 |         2.3540 |         3.0303 |               20.0 |          0.0
#       30.0 |         0.5714 |         2.8571 |               80.0 |          0.0
#       60.0 |         0.5714 |         2.8571 |               80.0 |          0.0
#      120.0 |         0.5714 |         2.8571 |               80.0 |          0.0
#      300.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#   last improvement [s]: median 22.9 | 90% 160.4 | max 248.4; remaining gap to the bound: mean 99.76 % | max 114.29 %
#
#   MaxCut ER-700-800, seed 70080004
#      T [s] | mean short [%] |  max short [%] |  reached final [%] |   no sol [%]
#        1.0 |       100.0000 |       100.0000 |                0.0 |          0.0
#        3.0 |         0.0153 |         0.0682 |               60.0 |          0.0
#        5.0 |         0.0153 |         0.0682 |               60.0 |          0.0
#       10.0 |         0.0153 |         0.0682 |               60.0 |          0.0
#       30.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#       60.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#      120.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#      300.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#   last improvement [s]: median 2.0 | 90% 10.6 | max 10.7; remaining gap to the bound: mean 78.40 % | max 79.64 %
#   (final incumbents 23349, 19870, 22886, 19214, 23475: below the degree greedy on the same instances, see
#    lns_init_comparison.jl)
#
#   MISP ER-700-800, seed 70080001
#      T [s] | mean short [%] |  max short [%] |  reached final [%] |   no sol [%]
#        1.0 |        26.4236 |        32.5581 |                0.0 |          0.0
#        3.0 |        26.4236 |        32.5581 |                0.0 |          0.0
#        5.0 |        26.4236 |        32.5581 |                0.0 |          0.0
#       10.0 |        22.3169 |        32.5581 |                0.0 |          0.0
#       30.0 |         9.7430 |        23.2558 |                0.0 |          0.0
#       60.0 |         1.3953 |         6.9767 |               80.0 |          0.0
#      120.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#      300.0 |         0.0000 |         0.0000 |              100.0 |          0.0
#   last improvement [s]: median 44.8 | 90% 77.3 | max 98.9; remaining gap to the bound: mean 259.66 % | max 274.36 %
#   (final incumbents 40, 39, 37, 40, 43: slightly above the greedy-start LNS with 3 s, see lns_init_comparison.jl)

using Printf
using Statistics

struct Trajectory
    name::String
    sols::Vector{Tuple{Float64,Float64}}      # (time, incumbent), improving solutions only
    final_time::Float64
    final_obj::Float64
    final_bound::Float64
end

function read_trajectory(path::AbstractString)
    sols = Tuple{Float64,Float64}[]
    final = nothing
    for line in Iterators.drop(eachline(path), 1)
        kind, t, inc, bnd = split(line, ',')
        if kind == "sol"
            push!(sols, (parse(Float64, t), parse(Float64, inc)))
        elseif kind == "final"
            final = (parse(Float64, t), parse(Float64, inc), parse(Float64, bnd))
        end
    end
    isnothing(final) && return nothing          # run did not finish (e.g. killed by Slurm)
    return Trajectory(basename(path), sols, final...)
end

"""Incumbent at time `T` (NaN if no solution was found by then)."""
function incumbent_at(tr::Trajectory, T::Float64)
    inc = NaN
    for (t, v) in tr.sols
        t <= T || break
        inc = v
    end
    return inc
end

"""Time of the last improvement of the incumbent."""
last_improvement(tr::Trajectory) = isempty(tr.sols) ? NaN : tr.sols[end][1]

quantile_or_nan(v, q) = isempty(v) ? NaN : quantile(v, q)

function main(args)
    isempty(args) && error("usage: analyze_gurobi_convergence.jl <dir> [checkpoints...]")
    dir = args[1]
    checkpoints = length(args) > 1 ? parse.(Float64, args[2:end]) : [1.0, 3, 5, 10, 30, 60, 120, 300, 600]

    files = filter(f -> startswith(f, "traj_") && endswith(f, ".csv"), readdir(dir))
    trajs = filter(!isnothing, [read_trajectory(joinpath(dir, f)) for f in files])
    n = length(trajs)
    n == 0 && error("no finished trajectories in $dir")
    length(files) > n && @warn "$(length(files) - n) trajectories have no final row (unfinished runs) and are ignored."

    t_max = maximum(tr.final_time for tr in trajs)
    @printf("%d instances, longest run %.1f s\n\n", n, t_max)
    println("Incumbent at T compared to the incumbent at the end of the run (maximization):")
    @printf("%8s | %14s | %14s | %18s | %12s\n", "T [s]", "mean short [%]", "max short [%]", "reached final [%]", "no sol [%]")
    println("-"^78)
    for T in checkpoints
        T > t_max * 1.001 && continue
        shortfalls = Float64[]
        reached = 0
        no_sol = 0
        for tr in trajs
            inc = incumbent_at(tr, T)
            if isnan(inc)
                no_sol += 1
                continue
            end
            push!(shortfalls, 100 * (tr.final_obj - inc) / tr.final_obj)
            reached += inc >= tr.final_obj - 1e-9
        end
        @printf("%8.1f | %14.4f | %14.4f | %18.1f | %12.1f\n", T,
                isempty(shortfalls) ? NaN : mean(shortfalls), isempty(shortfalls) ? NaN : maximum(shortfalls),
                100 * reached / n, 100 * no_sol / n)
    end

    lasts = filter(!isnan, [last_improvement(tr) for tr in trajs])
    gaps  = [100 * (tr.final_bound - tr.final_obj) / tr.final_obj for tr in trajs]
    println()
    @printf("Time of the last improvement [s]: median %.1f | 90%% %.1f | max %.1f\n",
            quantile_or_nan(lasts, 0.5), quantile_or_nan(lasts, 0.9), isempty(lasts) ? NaN : maximum(lasts))
    @printf("Remaining gap to the dual bound at the end [%%]: mean %.2f | max %.2f\n", mean(gaps), maximum(gaps))
    @printf("Proven optimal (gap ≤ 0.01%%): %d / %d\n", count(<=(0.01), gaps), n)
end

main(ARGS)
