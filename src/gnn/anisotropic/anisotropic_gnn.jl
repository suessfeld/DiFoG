using Flux
using Statistics
using ProgressMeter
using Zygote: ignore_derivatives
using Logging


export AnisotropicGNN

"""
Implementation of the anisotropic GNN proposed by Bresson and Laurent (2018).
"""
struct AnisotropicGNN{F} <: GNN
    config::GNNConfig
    layers::Vector{AnisotropicGNNLayer{F}}
end

# Necessary for Flux to track parameters
Flux.@functor AnisotropicGNN

# TODO: Check if layer dims are setup correctly.
"""
Constructor for AnisotropicGNN.

Args:
    config: GNNConfig containing in_dim num_layers, hidden_dim, and output_dim.    
"""
function AnisotropicGNN(config::GNNConfig)
    layers = Vector{AnisotropicGNNLayer{Float32}}()
    
    if config.num_layers == 1
        push!(layers, AnisotropicGNNLayer(config.in_dim, config.output_dim))
    else
        push!(layers, AnisotropicGNNLayer(config.in_dim, config.hidden_dim))
        
        for _ in 2:(config.num_layers - 1)
            push!(layers, AnisotropicGNNLayer(config.hidden_dim, config.hidden_dim))
        end
        
        push!(layers, AnisotropicGNNLayer(config.hidden_dim, config.output_dim))
    end
    
    return AnisotropicGNN(config, layers)
end

function AnisotropicGNN(config::GNNConfig, layers::Vector{AnisotropicGNNLayer{F}}) where F
    return AnisotropicGNN{F}(config, layers)
end

function (model::AnisotropicGNN)(x::AbstractMatrix, edge_index::AbstractMatrix)
    h = x
    for layer in model.layers
        h = layer(h, edge_index)
    end
    return h
end

function train(model::AnisotropicGNN, data)
    # TODO
end