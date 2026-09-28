# src/gnn/gnn_config.jl
#
# Configuration struct defining the hyperparameters and dimensionalities (layers, input/time/hidden/output
# dims) for the GNN models.

export GNNConfig
using Functors

"""
    GNNConfig

Defines the basic configuration, hyperparameters, and dimensionalities for the GNN models.

Fields:
    - 'num_layers': The number of message-passing layers.
    - 'in_dim': The width of the input node state.
    - 'time_dim': The width of the sinusoidal timestep embedding.
    - 'hidden_dim': The hidden dimension width.
    - 'output_dim': The number of output logits per node.
"""
struct GNNConfig
    num_layers::Int
    in_dim::Int
    time_dim::Int
    hidden_dim::Int
    output_dim::Int

    function GNNConfig(num_layers::Int, in_dim::Int, time_dim::Int, hidden_dim::Int, output_dim::Int)
        @assert num_layers >= 1 "num_layers must be ≥ 1"
        @assert in_dim >= 1 "in_dim (node state width) must be ≥ 1"
        @assert time_dim >= 2 "time_dim must be ≥ 2"
        @assert output_dim == 1 "output_dim must be 1 (one logit per node; the losses assume a (1, N) output)"
        return new(num_layers, in_dim, time_dim, hidden_dim, output_dim)
    end
end

Functors.@leaf GNNConfig

GNNConfig(; num_layers, in_dim, time_dim, hidden_dim, output_dim) =
    GNNConfig(num_layers, in_dim, time_dim, hidden_dim, output_dim)