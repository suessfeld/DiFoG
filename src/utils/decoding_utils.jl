# src/utils/decoding_utils.jl
#
# Utility functions for evaluating model predictions: hamming distances, heatmap decisiveness,
# correction distances, and the baseline heatmaps used as controls in every report.

using GraphNeuralNetworks
using Statistics
using Random

export hamming_distance, solution_distance, heatmap_decisiveness, correction_distance,
       degree_baseline_heatmap, random_baseline_score

"""
    decode_heatmap

A generic function that must be overridden by specific ProblemSolvers to decode predictions into solutions.
"""
function decode_heatmap end

"""
    evaluate

A generic function that must be overridden by specific ProblemSolvers to evaluate objective values.
"""
function evaluate end

"""
    degree_baseline_heatmap(solver::ProblemSolver, g::GNNGraph)

Generates a degree-heuristic baseline heatmap to be passed through the solver's decoder.

Parameters:
    - 'solver': The ProblemSolver instance.
    - 'g': The graph instance.
"""
function degree_baseline_heatmap end

"""
    random_baseline_score(solver::ProblemSolver, g::GNNGraph, k::Int; rng::AbstractRNG=Random.default_rng())

Calculates the average objective value of multiple uniformly random heatmaps passed through the problem's decoder.

Parameters:
    - 'solver': The ProblemSolver instance.
    - 'g': The graph structure.
    - 'k': The number of random heatmaps to generate and evaluate.
    - 'rng': The random number generator to use.
"""
function random_baseline_score(solver, g::GNNGraph, k::Int; rng::AbstractRNG=Random.default_rng())
    @assert k >= 1 "random_baseline_score: k must be ≥ 1"
    return mean(evaluate(solver, g, decode_heatmap(solver, g, rand(rng, Float32, g.num_nodes))) for _ in 1:k)
end

"""
    hamming_distance(a::AbstractArray, b::AbstractArray)

Calculates the elementwise Hamming distance between two equal-shaped binary or boolean arrays.

Parameters:
    - 'a': The first input array.
    - 'b': The second input array.
"""
function hamming_distance(a::AbstractArray, b::AbstractArray)
    @assert size(a) == size(b) "hamming_distance requires equal-sized inputs, got $(size(a)) vs $(size(b))"
    return sum(a .!= b)
end

"""
    solution_distance(a::AbstractVector, b::AbstractVector; symmetric::Bool=false)

Calculates the Hamming distance between two decoded solutions, optionally accounting for flip-invariant symmetries.

Parameters:
    - 'a': The first decoded solution vector.
    - 'b': The second decoded solution vector.
    - 'symmetric': Boolean flag to evaluate distance up to complement.
"""
function solution_distance(a::AbstractVector, b::AbstractVector; symmetric::Bool=false)
    d = hamming_distance(Bool.(a .> 0), Bool.(b .> 0))
    return symmetric ? min(d, length(a) - d) : d
end

"""
    heatmap_decisiveness(p::AbstractArray)

Calculates the mean absolute deviation of a heatmap from 0.5, returning a tuple of pseudo-entropy and confidence.

Parameters:
    - 'p': The heatmap array with values in [0,1].
"""
function heatmap_decisiveness(p::AbstractArray)
    deviation = @. abs(2.0f0 * Float32(p) - 1.0f0)
    confidence = mean(deviation)
    pseudo_entropy = 1.0f0 - confidence
    return pseudo_entropy, confidence
end

"""
    correction_distance(heatmap::AbstractVector, decoded_mask::AbstractVector; threshold::Float32=0.5f0)

Calculates the Hamming distance between a naive thresholded prediction and the final decoded feasible mask.

Parameters:
    - 'heatmap': The predicted heatmap vector.
    - 'decoded_mask': The final valid solution mask produced by the decoder.
    - 'threshold': The cutoff value used for the naive prediction.
"""
function correction_distance(heatmap::AbstractVector, decoded_mask::AbstractVector; threshold::Float32=0.5f0)
    naive = heatmap .> threshold
    return hamming_distance(naive, decoded_mask)
end