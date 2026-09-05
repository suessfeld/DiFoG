using Graphs

export MaxDegSolver

"""
A solver that solves the Maximum Degree Problem.
"""
struct MaxDegSolver <: ProblemSolver end

function (::MaxDegSolver)(g::Graphs.SimpleGraph)
    v = nv(g)
    
    degs = degree(g)
    max_deg = maximum(degs)
    max_deg_nodes = findall(x -> x == max_deg, degs)

    x_0 = zeros(Float32, 1, v)
    for idx in max_deg_nodes
        x_0[1, idx] = 1.0f0
    end
    
    return x_0
end

"""
    decode_heatmap(::MaxDegSolver, g::GNNGraph, heatmap::AbstractVector)

Simple decoding strategy for the MaxDeg problem. Returns the node with the highest probability value.

Parameters:
    - 'g': The graph representing the problem instance (unused, kept for signature consistency with other `decode_heatmap` methods).
    - 'heatmap': The predicted heatmap.
"""
function decode_heatmap(::MaxDegSolver, g::GNNGraph, heatmap::AbstractVector)
    degs = degree(g)
    max_deg = maximum(degs)
    k = count(==(max_deg), degs)

    mask = falses(g.num_nodes)
    top_k_idx = partialsortperm(vec(heatmap), 1:k, rev=true)
    mask[top_k_idx] .= true

    return mask
end