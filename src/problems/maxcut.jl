# src/problems/maxcut.jl
#
# Gurobi-based exact solver and heatmap decoding for the Maximum Cut (MAXCUT) problem, delegating the ILP formulation to NodeSelectionProblems.jl.

using Graphs
using JuMP

export MaxCutSolver

"""
    MaxCutSolver <: ProblemSolver
    
A solver that solves the Maximum Cut Problem with the usage of Gurobi.
"""
struct MaxCutSolver <: ProblemSolver
    time_limit::Float64
end

MaxCutSolver(; time_limit::Float64=0.5) = MaxCutSolver(time_limit)

"""
    (solver::MaxCutSolver)(g::Graphs.SimpleGraph)

Solves MaxCut for `g` via Gurobi. Delegates ILP formulation to `ILP!`/`NSPInstance{MAXCUT}`, and
session management/retry to `with_gurobi_retry` (see `gurobi_session.jl`) — this function only
describes how to build and read out a MaxCut solve; it knows nothing about sessions or retries.

Parameters:
    - 'g': The graph instance to solve MaxCut for.
"""
function (solver::MaxCutSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    x_0 = zeros(Float32, 1, n)
    gap = NaN

    result = with_gurobi_retry() do env
        inst = NSPInstance(MAXCUT(), g, n, ne(g), Ref{Union{Nothing, JuMP.Model}}(nothing), (;))
        ILP!(inst; silent=true, time_limit=solver.time_limit, warn_time_limit=false,
             env=env, threads=GRB_THREADS[])
    end

    if !isnothing(result)
        sol, status, rel_gap, solve_time = result
        for i in 1:sol.sel
            x_0[1, sol.x[i]] = 1.0f0
        end

        degs = degree(g)
        max_deg_node = argmax(degs)
        # cut is invariant under S ↔ V∖S, so the flip leaves the gap unchanged
        if x_0[1, max_deg_node] == 0.0f0
            x_0 .= 1.0f0 .- x_0
        end

        gap = Float64(rel_gap)
    end

    return x_0, gap
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