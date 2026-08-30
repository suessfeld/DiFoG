using GraphNeuralNetworks
using Statistics

export greedy_decode_mis, greedy_decode_maxcut, hamming_distance, heatmap_decisiveness, correction_distance

"""
    decode_heatmap

A generic function that must be overridden by the specific ProblemSolvers to decode problems.    
"""
function decode_heatmap end

"""
    hamming_distance(a::AbstractArray, b::AbstractArray)

Elementwise Hamming distance between two equal-shaped (binary or boolean) arrays.

Parameters:
    - 'a': Array a.
    - 'b': Array b.
"""
function hamming_distance(a::AbstractArray, b::AbstractArray)
    @assert size(a) == size(b) "hamming_distance requires equal-sized inputs, got $(size(a)) vs $(size(b))"
    return sum(a .!= b)
end

"""
    heatmap_decisiveness(p::AbstractArray)

Computes the mean absolute deviation of a heatmap p ∈ [0,1]^n from 0.5,
scaled to [0,1]: D̄ = mean(|2pᵢ - 1|).

Parameters:
    - 'p': The heatmap.
"""
function heatmap_decisiveness(p::AbstractArray)
    deviation = @. abs(2.0f0 * Float32(p) - 1.0f0)
    mean_deviation = mean(deviation)
    return mean_deviation, mean_deviation
end

"""
    correction_distance(heatmap::AbstractVector, decoded_mask::AbstractVector; threshold=0.5f0)

Measures how many nodes the greedy decoder had to flip relative to the naive
thresholded prediction in order to reach the final feasible greedy mask.
This is the raw Hamming distance between the naive prediction and the decoded
mask.

Parameters:
    - 'heatmap': The original, predicted heatmap.
    - 'decoded_mask': The result of the decoding.
    - 'threshold': The threshold defining when a probability is considered a 'pick'. Defaults to 0.5.
"""
function correction_distance(heatmap::AbstractVector, decoded_mask::AbstractVector; threshold::Float32=0.5f0)
    naive = heatmap .> threshold
    return hamming_distance(naive, decoded_mask)
end