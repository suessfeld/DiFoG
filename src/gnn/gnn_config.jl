export GNNConfig

"""
Represents a basic configuration of a GNN. 
"""
struct GNNConfig
    num_layers::Int
    in_dim::Int
    hidden_dim::Int
    output_dim::Int
end

function GNNConfig(; num_layers, in_dim, hidden_dim, output_dim)
    return GNNConfig(num_layers, in_dim, hidden_dim, output_dim)
end