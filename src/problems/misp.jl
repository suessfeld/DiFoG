# src/problems/misp.jl
#
# Gurobi-based exact solver for the Maximum Independent Set Problem (MISP) delegating ILP formulation to NodeSelectionProblems.jl, and implements greedy decoding based on the sun-23-difusco.pdf paper[cite: 1].

using Graphs
using JuMP
using Gurobi

export MISPSolver

"""
    MISPSolver <: ProblemSolver

A solver that solves the Maximum Independent Set Problem with the usage of Gurobi.
Delegates the actual ILP formulation to `NodeSelectionProblems.ILP!` (via `MIS`),
avoiding duplication of the MIS constraint/objective logic already defined there.
"""
struct MISPSolver <: ProblemSolver
    time_limit::Float64
end

MISPSolver(; time_limit::Float64=0.5) = MISPSolver(time_limit)

"""
    (solver::MISPSolver)(g::Graphs.SimpleGraph)

Solves MISP for `g` via Gurobi. Delegates ILP formulation to `ILP!`/`NSPInstance{MIS}`, and session
management/retry to `with_gurobi_retry` (see `gurobi_session.jl`) — this function only describes how
to build and read out a MISP solve; it knows nothing about sessions or retries.

Parameters:
    - 'g': The graph instance to solve MISP for.
"""
function (solver::MISPSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    x_0 = zeros(Float32, 1, n)
    gap = NaN

    result = with_gurobi_retry() do env
        inst = NSPInstance(MIS(0), g, n, ne(g), Ref{Union{Nothing, JuMP.Model}}(nothing), (;))
        ILP!(inst; silent=true, time_limit=solver.time_limit, warn_time_limit=false,
             env=env, threads=GRB_THREADS[])
    end

    if !isnothing(result)
        sol, status, rel_gap, solve_time = result
        for i in 1:sol.sel
            x_0[1, sol.x[i]] = 1.0f0
        end
        gap = Float64(rel_gap)
    end

    return x_0, gap
end

"""
    decode_heatmap(::MISPSolver, g::GNNGraph, heatmap::AbstractVector)

Greedy decoding strategy for MISP, as described in Section 3.5 of the
DIFUSCO paper: nodes are visited in descending order of predicted
confidence `heatmap[i]`, and greedily added to the independent set whenever
none of their neighbors are already selected.

Parameters:
    - 'g': The graph representing the problem instance.
    - 'heatmap': The per-node model confidence scores a_i ∈ [0,1].
"""
function decode_heatmap(::MISPSolver, g::GNNGraph, heatmap::AbstractVector)
    num_nodes = g.num_nodes

    sorted_nodes = sortperm(vec(heatmap), rev=true)

    in_set = falses(num_nodes)

    s, t = edge_index(g)
    adj = [Int[] for _ in 1:num_nodes]

    for i in 1:length(s)
        push!(adj[s[i]], t[i])
    end

    for node in sorted_nodes
        conflict = false
        for neighbor in adj[node]
            if in_set[neighbor]
                conflict = true
                break
            end
        end
        if !conflict
            in_set[node] = true
        end
    end

    return in_set
end

"""
    evaluate(::MISPSolver, g::GNNGraph, mask::AbstractVector)

MISP objective: cardinality of the selected (independent) set, i.e. Σᵢ maskᵢ.
"""
function evaluate(::MISPSolver, g::GNNGraph, mask::AbstractVector)
    return Float64(sum(mask))
end