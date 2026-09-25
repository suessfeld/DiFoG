# src/problems/misp.jl
#
# Gurobi-based exact solver for the Maximum Independent Set Problem (MISP) delegating ILP formulation to
# NodeSelectionProblems.jl, greedy decoding (DIFUSCO §3.5), and the degree baseline heatmap.

using Graphs
using JuMP
using Gurobi

export MISPSolver

"""
    MISPSolver <: ProblemSolver

A solver that exactly solves the Maximum Independent Set Problem using Gurobi, delegating the ILP formulation to NodeSelectionProblems.jl.

Fields:
    - 'time_limit': The maximum time limit allowed for the Gurobi solver.
"""
struct MISPSolver <: ProblemSolver
    time_limit::Float64
end

MISPSolver(; time_limit::Float64=0.5) = MISPSolver(time_limit)

"""
    (solver::MISPSolver)(g::Graphs.SimpleGraph)

Solves the Maximum Independent Set Problem for the given graph using Gurobi, handling session retries and failure states safely.

Parameters:
    - 'solver': The MISPSolver instance.
    - 'g': The graph instance to solve MISP for.
"""
function (solver::MISPSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    x_0 = zeros(Float32, 1, n)

    result = with_gurobi_retry() do env
        inst = NSPInstance(MIS(0), g, n, ne(g), Ref{Union{Nothing, JuMP.Model}}(nothing), (;))
        ILP!(inst; silent=true, time_limit=solver.time_limit, warn_time_limit=false,
             env=env, threads=GRB_THREADS[])
    end

    isnothing(result) && error("MISPSolver: Gurobi failed twice for a graph with $(nv(g)) nodes — refusing to emit an empty label.")

    sol, status, rel_gap, solve_time = result
    for i in 1:sol.sel
        x_0[1, sol.x[i]] = 1.0f0
    end
    gap = Float64(rel_gap)

    return x_0, gap
end

"""
    decode_heatmap(::MISPSolver, g::GNNGraph, heatmap::AbstractVector)

Applies a greedy decoding strategy by visiting nodes in descending order of predicted confidence and adding them to the independent set if no neighbors are already selected.

Parameters:
    - 'g': The graph representing the problem instance.
    - 'heatmap': A vector of per-node model confidence scores.
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
    degree_baseline_heatmap(::MISPSolver, g::GNNGraph)

Generates a baseline heatmap using the inverse node degrees, prompting the greedy decoder to visit nodes in ascending static degree order.

Parameters:
    - 'g': The graph structure.
"""
degree_baseline_heatmap(::MISPSolver, g::GNNGraph) = 1f0 ./ (1f0 .+ Float32.(degree(g)))

"""
    residue(solver::MISPSolver, g::GNNGraph)

Constructs the degraded solution for a residue source: the minimum-degree greedy independent set, i.e. the
degree baseline passed through the greedy decoder. It is feasible and maximal.

Parameters:
    - 'g': The graph structure.
"""
residue(solver::MISPSolver, g::GNNGraph) =
    Float32.(decode_heatmap(solver, g, degree_baseline_heatmap(solver, g)))

"""
    evaluate(::MISPSolver, g::GNNGraph, mask::AbstractVector)

Evaluates the Maximum Independent Set objective value by calculating the cardinality of the selected node set.

Parameters:
    - 'g': The graph structure.
    - 'mask': A binary vector indicating the selected nodes.
"""
function evaluate(::MISPSolver, g::GNNGraph, mask::AbstractVector)
    return Float64(sum(mask))
end