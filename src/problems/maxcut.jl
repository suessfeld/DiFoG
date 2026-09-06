# src/problems/maxcut.jl
#
# Gurobi-based exact solver and heatmap decoding for the Maximum Cut (MAXCUT) problem, delegating the ILP formulation to NodeSelectionProblems.jl.

using Graphs
using JuMP

export MaxCutSolver

"""
A solver that solves the Maximum Cut Problem with the usage of Gurobi.
"""
struct MaxCutSolver <: ProblemSolver
    time_limit::Float64
end

MaxCutSolver(; time_limit::Float64=0.5) = MaxCutSolver(time_limit)

"""
    (solver::MaxCutSolver)(g::Graphs.SimpleGraph)

Solves MaxCut for `g` via Gurobi. Delegates the ILP formulation
to `ILP!`/`NSPInstance{MAXCUT}`.

Retries once with a fresh `Gurobi.Env()` if the WLS connection drops mid-solve;
a poisoned environment is discarded and replaced in the background
(`GRB_ENV_POOL`) rather than returned to the pool.

Parameters:
    - 'g': The graph instance to solve MaxCut for.
"""
function (solver::MaxCutSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    env = take!(GRB_ENV_POOL)
    x_0 = zeros(Float32, 1, n)
    is_poisoned = false

    function build_and_solve(current_env)
        inst = NSPInstance(
            MAXCUT(), g, n, ne(g),
            Ref{Union{Nothing, JuMP.Model}}(nothing),
            (;),
        )
        return ILP!(inst; silent=true, time_limit=solver.time_limit,
                     warn_time_limit=false, env=current_env)
    end

    function apply_solution!(sol)
        for i in 1:sol.sel
            x_0[1, sol.x[i]] = 1.0f0
        end
        
        degs = degree(g)
        max_deg_node = argmax(degs)
        
        if x_0[1, max_deg_node] == 0.0f0
            x_0 .= 1.0f0 .- x_0
        end
    end

    try
        sol, status, rel_gap, solve_time = build_and_solve(env)
        apply_solution!(sol)
    catch opt_e
        @warn "Gurobi Token/Network Dropped. Discarding environment and attempting recovery..." exception=opt_e
        is_poisoned = true

        try
            new_env = Gurobi.Env()
            sol, status, rel_gap, solve_time = build_and_solve(new_env)
            apply_solution!(sol)
            env = new_env
            is_poisoned = false
        catch recovery_e
            @error "WLS recovery failed. Network connection is completely offline." exception=recovery_e
        end
    finally
        if !is_poisoned
            put!(GRB_ENV_POOL, env)
        else
            Threads.@spawn begin
                sleep(5)
                try
                    recovered_env = Gurobi.Env()
                    put!(GRB_ENV_POOL, recovered_env)
                catch e
                    sleep(10)
                    try
                        recovered_env = Gurobi.Env()
                        put!(GRB_ENV_POOL, recovered_env)
                    catch second_e
                    end
                end
            end
        end
    end

    return x_0
end

"""
    decode_heatmap(::MaxCutSolver, g::GNNGraph, heatmap::AbstractVector)

Simple decoding strategy for maxcut. Divides the set into two subsets, one set includes all values p > 0.5, the other one p < 0.5.

Parameters:
    - 'g': The graph representing the problem instance (unused, kept for signature consistency with other `decode_heatmap` methods).
    - 'heatmap': The predicted heatmap.
"""
function decode_heatmap(::MaxCutSolver, g::GNNGraph, heatmap::AbstractVector)
    return vec(heatmap) .> 0.5f0
end

"""
    evaluate(::MaxCutSolver, g::GNNGraph, mask::AbstractVector)

MAXCUT objective: |{(u,v) ∈ E : mask[u] ≠ mask[v]}|, i.e. the number of edges
crossing the cut induced by `mask`. `edge_index(g)` stores each undirected
edge twice (once per direction), hence the `÷ 2`.
"""
function evaluate(::MaxCutSolver, g::GNNGraph, mask::AbstractVector)
    s, t = edge_index(g)
    selected = Bool.(vec(mask))
    cut = 0
    @inbounds for i in eachindex(s)
        cut += selected[s[i]] != selected[t[i]] ? 1 : 0
    end
    return Float64(cut ÷ 2)
end