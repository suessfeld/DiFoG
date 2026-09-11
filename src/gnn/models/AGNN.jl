# src/gnn/models/AGNN.jl
#
# Implements the Anisotropic Graph Neural Network (AGNN) backbone, handling message passing and feature extraction for both nodes and edges.

using Flux
using Statistics
using GraphNeuralNetworks
using NNlib
using CUDA
using Zygote: ignore_derivatives

export AGNN

"""
    OutputHead

Final node-wise prediction head: BatchNorm + ReLU + Dense.
"""
struct OutputHead
    bn     :: BatchNorm
    linear :: Dense
end
Flux.@layer OutputHead

OutputHead(d::Int, output_dim::Int) = OutputHead(
    BatchNorm(d, track_stats=false),
    Dense(d, output_dim)
)

"""
    (m::OutputHead)(h::AbstractMatrix)

Maps hidden node features to output logits.
"""
(m::OutputHead)(h::AbstractMatrix) = m.linear(relu.(m.bn(h)))

"""
    AGNNLayer

One anisotropic message-passing layer, updating both node (`h`) and edge (`e`) features.
"""
struct AGNNLayer
    U      :: Dense
    V      :: Dense
    A      :: Dense
    B      :: Dense
    C      :: Dense
    norm_h :: BatchNorm
    norm_e :: BatchNorm
    mlp_e  :: Chain
end
Flux.@layer AGNNLayer

AGNNLayer(d::Int) = AGNNLayer(
    Dense(d, d),
    Dense(d, d),
    Dense(d, d),
    Dense(d, d),
    Dense(d, d),
    BatchNorm(d, track_stats=false),
    BatchNorm(d, track_stats=false),
    Chain(Dense(d, d, relu), Dense(d, d)),
)

"""
    (layer::AGNNLayer)(g::GNNGraph, h::AbstractMatrix, e::AbstractMatrix)

Computes one anisotropic node/edge update step: gated message passing for `h`,
edge-gate MLP for `e`. Returns `(h_delta, e_delta)`.
"""
function (layer::AGNNLayer)(g::GNNGraph, h::AbstractMatrix, e::AbstractMatrix)
    s, t = edge_index(g)

    Uh = layer.U(h)
    Vh = layer.V(h)
    Ah = layer.A(h)
    Bh = layer.B(h)
    Ce = layer.C(e)

    e_hat = NNlib.gather(Ah, s) .+ NNlib.gather(Bh, t) .+ Ce

    messages = sigmoid.(e_hat) .* NNlib.gather(Vh, s)
    agg = NNlib.scatter(+, messages, t; dstsize=(size(h, 1), g.num_nodes))

    h_delta = relu.(layer.norm_h(Uh .+ agg))
    e_delta = layer.mlp_e(layer.norm_e(e_hat))

    return h_delta, e_delta
end

"""
    AGNN <: GNN

Implementation of the Ansiotropic Graph Neural Network proposed by Laurent & Bresson 2018.
"""
struct AGNN <: GNN
    config      :: GNNConfig
    feature_dim :: Int
    node_embed  :: Dense
    layers      :: Vector{AGNNLayer}
    time_layers :: Vector{Chain}
    output_head :: OutputHead
end
Flux.@layer AGNN

"""
    AGNN(config::GNNConfig, feature_dim::Int=1)

Builds an `AGNN` with `config.num_layers` `AGNNLayer`s, sized from `config`.
`feature_dim` is the raw node-state width; the remainder of `config.in_dim`
is treated as timestep-embedding width.
"""
function AGNN(config::GNNConfig, feature_dim::Int=1)
    d        = config.hidden_dim
    time_dim = config.in_dim - feature_dim
    L        = config.num_layers

    @assert time_dim > 0 "in_dim ($(config.in_dim)) must be > feature_dim ($feature_dim)"

    return AGNN(
        config,
        feature_dim,
        Dense(feature_dim, d, relu),
        [AGNNLayer(d) for _ in 1:L],
        [Chain(Dense(time_dim, d, relu), Dense(d, d)) for _ in 1:L],
        OutputHead(d, config.output_dim),
    )
end

"""
    (model::AGNN)(g::GNNGraph, x_input::AbstractMatrix)

Forward pass: splits `x_input` into raw state and timestep embedding, embeds
nodes, runs `L` message-passing layers (feeding the time embedding into each
edge update), and returns per-node output logits.
"""
function (model::AGNN)(g::GNNGraph, x_input::AbstractMatrix)
    fd = model.feature_dim
    x_t   = x_input[end-fd+1:end, :]
    t_emb = x_input[1:end-fd, :]

    h = model.node_embed(x_t)
    e = ignore_derivatives() do
        CUDA.zeros(Float32, model.config.hidden_dim, g.num_edges)
    end

    s, _ = edge_index(g)
    t_emb_edges = t_emb[:, s]

    for i in eachindex(model.layers)
        h_in, e_in = h, e
        h_delta, e_delta = model.layers[i](g, h, e)
        h = h_in .+ h_delta
        e = e_in .+ e_delta .+ model.time_layers[i](t_emb_edges)
    end

    return model.output_head(h)
end