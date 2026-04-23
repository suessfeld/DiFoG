using Flux
using Statistics
using ProgressMeter
using Zygote: ignore_derivatives
using Logging
using GraphNeuralNetworks

export AnisotropicGNNBasic

"""
Implementation of the anisotropic GNN proposed by Bresson and Laurent (2018).
"""
struct AnisotropicGNNBasic <: GNN
    config::GNNConfig
    layers::GNNChain
end

Flux.@layer AnisotropicGNNBasic

"""
    AnisotropicGNNBasic(config::GNNConfig)

Constructor for a basic (firs version) AnisotropicGNN.

Parameters:
    - 'config' : GNNConfig 
"""
function AnisotropicGNNBasic(config::GNNConfig)
    if config.num_layers == 1
        chain = GNNChain(ResGatedGraphConv(config.in_dim => config.output_dim, relu))
    else
        layers = Any[ResGatedGraphConv(config.in_dim => config.hidden_dim, relu)]
        
        for _ in 2:(config.num_layers - 1)
            push!(layers, ResGatedGraphConv(config.hidden_dim => config.hidden_dim, relu))
        end
        
        push!(layers, ResGatedGraphConv(config.hidden_dim => config.output_dim))
        
        chain = GNNChain(Tuple(layers)...)
    end
    
    return AnisotropicGNNBasic(config, chain)
end

function (model::AnisotropicGNNBasic)(g::GNNGraph, x::AbstractMatrix)
    return model.layers(g, x) 
end