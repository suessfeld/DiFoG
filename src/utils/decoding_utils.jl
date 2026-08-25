# src/utils/decoding_utils.jl

using GraphNeuralNetworks
using Statistics

export greedy_decode_mis, greedy_decode_maxcut, hamming_distance, heatmap_decisiveness, correction_distance

"""
    greedy_decode_mis(g::GNNGraph, a::AbstractVector)

Implements a simple, greedy decoding strategy for the MIS problem.
as described in Section 3.5 of the DIFUSCO paper.

Parameters:
    - 'g': The graph representing the problem instance.
    - 'a': The heatmap scores denoting the confidence of each node (a_i).
"""
function greedy_decode_mis(g::GNNGraph, a::AbstractVector)
    num_nodes = g.num_nodes

    sorted_nodes = sortperm(vec(a), rev=true) # first sort nodes

    in_set = falses(num_nodes)

    s, t = edge_index(g)
    adj = [Int[] for _ in 1:num_nodes]

    for i in 1:length(s)
        push!(adj[s[i]], t[i])
    end

    for node in sorted_nodes
        conflict = false

        # check if any neighbor is already in the independent set
        for neighbor in adj[node]
            if in_set[neighbor]
                conflict = true
                break
            end
        end

        # if no neighbors are in the set, it's safe to add this node
        if !conflict
            in_set[node] = true
        end
    end

    return in_set
end

function greedy_decode_maxcut(g::GNNGraph, a::AbstractVector)
    num_nodes = g.num_nodes
    conf = abs.(vec(a) .- 0.5f0)
    sorted_nodes = sortperm(conf, rev=true)
    in_set = fill(-1, num_nodes)
    s, t = edge_index(g)
    adj = [Int[] for _ in 1:num_nodes]
    for i in 1:length(s)
        push!(adj[s[i]], t[i])
    end
    for node in sorted_nodes
        c0 = 0
        c1 = 0
        for neighbor in adj[node]
            if in_set[neighbor] == 0
                c0 += 1
            elseif in_set[neighbor] == 1
                c1 += 1
            end
        end
        if c0 > c1
            in_set[node] = 1
        elseif c1 > c0
            in_set[node] = 0
        else
            in_set[node] = vec(a)[node] > 0.5f0 ? 1 : 0
        end
    end
    return in_set .== 1
end

"""
    hamming_distance(a::AbstractArray, b::AbstractArray)

Elementwise Hamming distance between two equal-shaped (binary or boolean) arrays.
"""
function hamming_distance(a::AbstractArray, b::AbstractArray)
    @assert size(a) == size(b) "hamming_distance requires equal-sized inputs, got $(size(a)) vs $(size(b))"
    return sum(a .!= b)
end

"""
    heatmap_decisiveness(p::AbstractArray; eps=1f-7)

Computes the mean binary entropy of a heatmap p ∈ [0,1]^n:

    H̄ = mean( -pᵢ log(pᵢ) - (1-pᵢ) log(1-pᵢ) )

H̄ → 0 as the heatmap saturates towards {0,1} (maximally decisive),
H̄ → ln(2) as pᵢ → 0.5 everywhere (maximally uncertain).

Returns (mean_entropy, confidence) where
    confidence = 1 - H̄ / ln(2) ∈ [0,1],  1 = fully decisive.
"""
function heatmap_decisiveness(p::AbstractArray; eps::Float32=1f-7)
    p_clamped = clamp.(Float32.(p), eps, 1.0f0 - eps)
    entropy = @. -p_clamped * log(p_clamped) - (1.0f0 - p_clamped) * log(1.0f0 - p_clamped)
    mean_entropy = mean(entropy)
    confidence = 1.0f0 - mean_entropy / Float32(log(2))
    return mean_entropy, confidence
end

"""
    correction_distance(heatmap::AbstractVector, decoded_mask::AbstractVector; threshold=0.5f0)

Measures how many nodes the greedy decoder had to flip relative to the naive
thresholded prediction 1[pᵢ > threshold] (which is generally infeasible w.r.t.
the independence constraint) in order to reach the final feasible greedy mask.
This is the raw Hamming distance between the naive prediction and the decoded
mask — a proxy for how much "correction work" greedy decoding performed.
"""
function correction_distance(heatmap::AbstractVector, decoded_mask::AbstractVector; threshold::Float32=0.5f0)
    naive = heatmap .> threshold
    return hamming_distance(naive, decoded_mask)
end