using Flux
using Statistics
using ProgressMeter
using Zygote: ignore_derivatives
using Logging
using GraphNeuralNetworks

export AnisotropicGNN

"""
Implementation of the anisotropic GNN proposed by Bresson and Laurent (2018).
"""
struct AnisotropicGNN <: GNN
    config::GNNConfig
    layers::GNNChain
end

Flux.@layer AnisotropicGNN

"""
    AnisotropicGNN(config::GNNConfig)

Constructor for AnisotropicGNN.

Parameters:
    - 'config' : GNNConfig 
"""
function AnisotropicGNN(config::GNNConfig)
    layers = []
    
    if config.num_layers == 1
        push!(layers, ResGatedGraphConv(config.in_dim => config.output_dim, relu))
    else
        push!(layers, ResGatedGraphConv(config.in_dim => config.hidden_dim, relu))
        
        for _ in 2:(config.num_layers - 1)
            push!(layers, ResGatedGraphConv(config.hidden_dim => config.hidden_dim, relu))
        end
        
        push!(layers, ResGatedGraphConv(config.hidden_dim => config.output_dim))
    end
    
    return AnisotropicGNN(config, GNNChain(layers...))
end

function (model::AnisotropicGNN)(g::GNNGraph, x::AbstractMatrix)
    return model.layers(g, x) 
end