using Graphs
using JuMP

export MaxCutSolver

"""
    MaxCutSolver <: ProblemSolver

A solver that solves the Maximum Cut Problem using Gurobi.

Fields:
    - 'time_limit': The maximum time limit allowed for the Gurobi solver.
"""
struct MaxCutSolver <: ProblemSolver
    time_limit::Float64
end

MaxCutSolver(; time_limit::Float64=0.5) = MaxCutSolver(time_limit)

"""
    label_symmetric(::MaxCutSolver)

Indicates whether the training loss should be flip-invariant, as a cut and its complement represent the exact same solution.
"""
label_symmetric(::MaxCutSolver) = true

"""
    (solver::MaxCutSolver)(g::Graphs.SimpleGraph)

Solves the Maximum Cut problem for the given graph using Gurobi, delegating the ILP formulation and handling session retries.

Parameters:
    - 'solver': The MaxCutSolver instance.
    - 'g': The graph instance to solve MaxCut for.
"""
function (solver::MaxCutSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    x_0 = zeros(Float32, 1, n)

    result = with_gurobi_retry() do env
        inst = NSPInstance(MAXCUT(), g, n, ne(g), Ref{Union{Nothing, JuMP.Model}}(nothing), (;))
        ILP!(inst; silent=true, time_limit=solver.time_limit, warn_time_limit=false,
             env=env, threads=GRB_THREADS[])
    end

    isnothing(result) && error("MaxCutSolver: Gurobi failed twice for a graph with $(nv(g)) nodes — refusing to emit an empty label.")

    sol, status, rel_gap, solve_time = result
    for i in 1:sol.sel
        x_0[1, sol.x[i]] = 1.0f0
    end

    degs = degree(g)
    max_deg_node = argmax(degs)
    
    if x_0[1, max_deg_node] == 0.0f0
        x_0 .= 1.0f0 .- x_0
    end

    gap = Float64(rel_gap)

    return x_0, gap
end

"""
    decode_heatmap(::MaxCutSolver, g::GNNGraph, heatmap::AbstractVector)

Decodes the predicted heatmap into a binary partition using a 0.5 threshold.

Parameters:
    - 'g': The graph structure.
    - 'heatmap': The predicted heatmap vector to be decoded.
"""
function decode_heatmap(::MaxCutSolver, g::GNNGraph, heatmap::AbstractVector)
    return vec(heatmap) .> 0.5f0
end

"""
    degree_baseline_heatmap(::MaxCutSolver, g::GNNGraph)

Calculates a baseline heatmap by visiting nodes in descending degree order and placing each node on the side with fewer already-placed neighbors.

Parameters:
    - 'g': The graph structure.
"""
function degree_baseline_heatmap(::MaxCutSolver, g::GNNGraph)
    n = g.num_nodes
    s, t = edge_index(g)
    adj = [Int[] for _ in 1:n]
    for i in eachindex(s)
        push!(adj[s[i]], t[i])
    end

    side = zeros(Int8, n)
    for v in sortperm(length.(adj), rev=true)
        in_s  = count(u -> side[u] == 1, adj[v])
        in_vs = count(u -> side[u] == -1, adj[v])
        side[v] = in_s <= in_vs ? Int8(1) : Int8(-1)
    end
    return Float32.(side .== 1)
end

"""
    evaluate(::MaxCutSolver, g::GNNGraph, mask::AbstractVector)

Evaluates the Maximum Cut objective value for a given partition mask.

Parameters:
    - 'g': The graph structure.
    - 'mask': A binary vector representing the node partition.
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