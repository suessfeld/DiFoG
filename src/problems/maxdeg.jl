# src/problems/maxdeg.jl
#
# Solver and heatmap decoding strategy for the Maximum Degree (MaxDeg) problem, operating in conjunction
# with the exact problem definitions in NodeSelectionProblems.jl.

using Graphs

export MaxDegSolver

"""
    MaxDegSolver <: ProblemSolver

A solver that calculates the exact solution for the Maximum Degree problem.
"""
struct MaxDegSolver <: ProblemSolver end

"""
    (solver::MaxDegSolver)(g::Graphs.SimpleGraph)

Solves the Maximum Degree problem exactly by identifying the nodes with the highest degree, resulting in a gap of 0.

Parameters:
    - 'solver': The MaxDegSolver instance.
    - 'g': The graph instance to solve.
"""
function (solver::MaxDegSolver)(g::Graphs.SimpleGraph)
    v = nv(g)

    degs = degree(g)
    max_deg = maximum(degs)
    max_deg_nodes = findall(x -> x == max_deg, degs)

    x_0 = zeros(Float32, 1, v)
    for idx in max_deg_nodes
        x_0[1, idx] = 1.0f0
    end

    return x_0, 0.0
end

"""
    decode_heatmap(::MaxDegSolver, g::GNNGraph, heatmap::AbstractVector)

Decodes the heatmap by selecting the top k nodes with the highest predicted values, where k is the true number of maximum-degree nodes.

Parameters:
    - 'g': The graph representing the problem instance.
    - 'heatmap': The predicted heatmap vector.
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
    degree_baseline_heatmap(::MaxDegSolver, g::GNNGraph)

Generates a baseline heatmap using the node degrees directly, serving as a sanity check that decodes to the exact solution.

Parameters:
    - 'g': The graph structure.
"""
degree_baseline_heatmap(::MaxDegSolver, g::GNNGraph) = Float32.(degree(g))

"""
    evaluate(::MaxDegSolver, g::GNNGraph, mask::AbstractVector)

Evaluates the Maximum Degree objective by summing the degrees of the selected nodes.

Parameters:
    - 'g': The graph structure.
    - 'mask': A binary vector indicating the selected nodes.
"""
function evaluate(::MaxDegSolver, g::GNNGraph, mask::AbstractVector)
    degs = degree(g)
    selected = Bool.(vec(mask))
    return Float64(sum(degs[selected]; init=0))
end