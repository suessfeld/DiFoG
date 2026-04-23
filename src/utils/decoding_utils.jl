# src/utils/decoding_utils.jl

using GraphNeuralNetworks

export greedy_decode_mis

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