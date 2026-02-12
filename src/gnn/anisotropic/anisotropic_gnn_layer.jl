using Flux
using NNlib
using Statistics
using ProgressMeter
using Logging

export AnisotropicGNNLayer

struct AnisotropicGNNLayer{F} <: Layer
    U::Matrix{F}
    V::Matrix{F}
    A::Matrix{F}
    B::Matrix{F}
end

function AnisotropicGNNLayer(in_dim::Int, out_dim::Int)
    # Init with glot - TODO: Maybe change strategy
    AnisotropicGNNLayer(
        Flux.glorot_uniform(out_dim, in_dim),
        Flux.glorot_uniform(out_dim, in_dim),
        Flux.glorot_uniform(out_dim, in_dim),
        Flux.glorot_uniform(out_dim, in_dim)
    )
end

# Enable Flux to see the parameters
Flux.@layer AnisotropicGNNLayer

"""
Forward pass for this GNN.

    Original formula: RELU(U*h_i + sum_{j->i} η_ij ⊙ (V*h_j)) where
    η_ij ... sigmoid(A*h_i + B*h_j).
    U, V, A, B ... parameters
    h ... hidden state of nodes
"""
function (model::AnisotropicGNNLayer)(h::AbstractMatrix, edge_index::AbstractMatrix)  
    # Batch-wise computations so we dont have to for-loop
    h_U = model.U * h
    h_V = model.V * h
    h_A = model.A * h
    h_B = model.B * h

    # Get indices (ensure they are integers for indexing)
    src_nodes = Int.(edge_index[1, :])
    dst_nodes = Int.(edge_index[2, :])
    
    # Gate calculation: η_ij = σ(A*h_i + B*h_j)
    gate_inputs = h_A[:, dst_nodes] .+ h_B[:, src_nodes]
    η = sigmoid.(gate_inputs)

    # Message: η_ij ⊙ (V * h_j)
    # Using 'src_nodes' to pull features from neighbors
    messages = η .* h_V[:, src_nodes]

    # Aggregate
    aggregation = NNlib.scatter(+, messages, dst_nodes, dstsize=size(h))

    # ReLU in the end
    return relu.(h_U .+ aggregation) 
end