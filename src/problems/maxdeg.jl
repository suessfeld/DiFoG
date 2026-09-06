# src/problems/maxdeg.jl
#
# Solver and heatmap decoding strategy for the Maximum Degree (MaxDeg) problem, operating in conjunction with the exact problem definitions in NodeSelectionProblems.jl.

using Graphs

export MaxDegSolver

"""
    MaxDegSolver <: ProblemSolver
    
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

"""
    evaluate(::MaxDegSolver, g::GNNGraph, mask::AbstractVector)

MaxDeg objective: Σᵢ degree(i) for i ∈ selected set, matching
`MHLib.calc_objective(sol::NSPSolution{MaxDeg})`.
"""
function evaluate(::MaxDegSolver, g::GNNGraph, mask::AbstractVector)
    degs = degree(g)
    selected = Bool.(vec(mask))
    return Float64(sum(degs[selected]; init=0))
end