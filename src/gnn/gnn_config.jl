# src/gnn/gnn_config.jl
#
# Configuration struct defining the hyperparameters and dimensionalities (layers, input/hidden/output dims) for the GNN models.

export GNNConfig
using Functors

"""
    GNNConfig
    
Represents a basic configuration of a GNN. 
"""
struct GNNConfig
    num_layers::Int
    in_dim::Int
    hidden_dim::Int
    output_dim::Int
end

Functors.@leaf GNNConfig

function GNNConfig(; num_layers, in_dim, hidden_dim, output_dim)
    return GNNConfig(num_layers, in_dim, hidden_dim, output_dim)
end