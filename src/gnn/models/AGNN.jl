using Flux
using Statistics
using GraphNeuralNetworks
using NNlib
using Zygote: ignore_derivatives

export AGNN

# Normalization: LayerNorm (per node / edge over its features), as in the reference implementation of DIFUSCO and
# EFLOCO (GNNEncoder with the default norm="layer"), although the DIFUSCO paper names BatchNorm. BatchNorm would
# normalize over the whole batch, whose graphs are at different noise levels during training (one t per graph)
# but at one shared level during inference, and would make the results depend on the batch size and on the
# co-batched graphs.

"""
    OutputHead

Final node-wise prediction head applying layer normalization, ReLU activation, and a dense layer.

Fields:
    - 'norm': The layer normalization.
    - 'linear': The dense layer for output projection.
"""
struct OutputHead
    norm   :: LayerNorm
    linear :: Dense
end
Flux.@layer OutputHead

OutputHead(d::Int, output_dim::Int) = OutputHead(
    LayerNorm(d),
    Dense(d, output_dim)
)

"""
    (m::OutputHead)(h::AbstractMatrix)

Maps hidden node features to output logits.

Parameters:
    - 'm': The OutputHead instance.
    - 'h': The hidden node features matrix.
"""
(m::OutputHead)(h::AbstractMatrix) = m.linear(relu.(m.norm(h)))

"""
    AGNNLayer

Constructs an anisotropic message-passing layer for updating node and edge features.

Fields:
    - 'U': Dense layer for node feature transformation.
    - 'V': Dense layer for neighbor feature transformation.
    - 'A': Dense layer for edge feature computation (source).
    - 'B': Dense layer for edge feature computation (target).
    - 'C': Dense layer for edge feature computation (edge).
    - 'norm_h': Layer normalization for node features.
    - 'norm_e': Layer normalization for edge features.
    - 'mlp_e': Multi-layer perceptron for final edge feature update.
"""
struct AGNNLayer
    U      :: Dense
    V      :: Dense
    A      :: Dense
    B      :: Dense
    C      :: Dense
    norm_h :: LayerNorm
    norm_e :: LayerNorm
    mlp_e  :: Chain
end
Flux.@layer AGNNLayer

AGNNLayer(d::Int) = AGNNLayer(
    Dense(d, d),
    Dense(d, d),
    Dense(d, d),
    Dense(d, d),
    Dense(d, d),
    LayerNorm(d),
    LayerNorm(d),
    Chain(Dense(d, d, relu), Dense(d, d)),
)

"""
    (layer::AGNNLayer)(g::GNNGraph, h::AbstractMatrix, e::AbstractMatrix)

Performs a single anisotropic update step for node and edge features.

Parameters:
    - 'layer': The AGNNLayer instance.
    - 'g': The graph structure.
    - 'h': The current node features matrix.
    - 'e': The current edge features matrix.
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

Represents the Anisotropic Graph Neural Network backbone for message passing and feature extraction.

Fields:
    - 'config': The configuration object for the GNN.
    - 'feature_dim': The dimensionality of the input node features.
    - 'node_embed': The dense layer for initial node embedding.
    - 'layers': A vector of anisotropic message-passing layers.
    - 'time_layers': A vector of chains processing the timestep embeddings.
    - 'output_head': The final output prediction head.
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
    AGNN(config::GNNConfig)

Builds an AGNN model based on the provided configuration.

Parameters:
    - 'config': The configuration object specifying layer counts and dimensions.
"""
function AGNN(config::GNNConfig)
    d = config.hidden_dim
    L = config.num_layers

    return AGNN(
        config,
        config.in_dim,
        Dense(config.in_dim, d, relu),
        [AGNNLayer(d) for _ in 1:L],
        [Chain(Dense(config.time_dim, d, relu), Dense(d, d)) for _ in 1:L],
        OutputHead(d, config.output_dim),
    )
end

"""
    (model::AGNN)(g::GNNGraph, x_t::AbstractMatrix, t_emb::AbstractMatrix)

Executes the forward pass of the network, updating node and edge states with time embeddings across all layers.

Parameters:
    - 'model': The AGNN model instance.
    - 'g': The graph structure containing batch information.
    - 'x_t': The node state matrix at the current timestep.
    - 't_emb': The per-graph timestep embedding matrix.
"""
function (model::AGNN)(g::GNNGraph, x_t::AbstractMatrix, t_emb::AbstractMatrix)
    @assert size(x_t, 1) == model.feature_dim "node state width $(size(x_t, 1)) ≠ in_dim $(model.feature_dim)"
    @assert size(t_emb, 1) == model.config.time_dim "time embedding width $(size(t_emb, 1)) ≠ time_dim $(model.config.time_dim)"

    h = model.node_embed(x_t)
    
    e = ignore_derivatives() do
        fill!(similar(h, model.config.hidden_dim, g.num_edges), 0f0)
    end

    edge_graph = ignore_derivatives() do
        isnothing(g.graph_indicator) ? nothing : g.graph_indicator[edge_index(g)[1]]
    end

    for i in eachindex(model.layers)
        h_in, e_in = h, e
        h_delta, e_delta = model.layers[i](g, h, e)
        
        h = h_in .+ h_delta
        
        t_graph = model.time_layers[i](t_emb)
        t_edge = isnothing(edge_graph) ? t_graph : NNlib.gather(t_graph, edge_graph)
        
        e = e_in .+ e_delta .+ t_edge
    end

    return model.output_head(h)
end