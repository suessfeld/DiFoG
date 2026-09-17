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
#GR TODO Claude meint hier: Time MLP evaluated per edge: `t_emb_edges = t_emb[:, s]` then `time_layers[i](…)` over all E
#   columns in each of 12 layers (`AGNN.jl:147-155`) although there are only `num_graphs` distinct
#   columns. For 16 × 350-node ER graphs (~150k directed edges) ~90 % of the time-MLP FLOPs, forward and
#   backward, are redundant. Compute on `(time_dim, num_graphs)` and gather by `graph_indicator[s]`.
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


#GR TODO Claude zeigt in Bezug auf die GPU/Flux folgende teils sehr wesentliche Probleme in Bezug auf Effizienz auf: Bitte durchdenken und auch hier kommentieren, wie Sie das sehen bzw. was geändert wurde:
# - **P1** Time MLP evaluated per edge: `t_emb_edges = t_emb[:, s]` then `time_layers[i](…)` over all E
#   columns in each of 12 layers (`AGNN.jl:147-155`) although there are only `num_graphs` distinct
#   columns. For 16 × 350-node ER graphs (~150k directed edges) ~90 % of the time-MLP FLOPs, forward and
#   backward, are redundant. Compute on `(time_dim, num_graphs)` and gather by `graph_indicator[s]`.
# - **P2** Schedule constants recomputed and re-uploaded per call: `get_alphas` (CPU `range` + `cumprod`
#   + `|> gpu`) once per training batch in `q` and twice per denoising step in `posterior_prob`
#   (`bernoulli_diffusion.jl:68,139-146`), each a sync point; `get_timestep_embedding` rebuilds and uploads
#   `freqs` every forward pass and allocates `hcat` + `permutedims` (`diffusion_model.jl:254-263`).
# - **P3** `_mat(g::AbstractVector) = map(_mat, g)` (`diffusion_model.jl:166`) matches every `CuVector`
#   bias/γ/β gradient: an extra kernel launch and copy per array per step (~140 arrays), while matrices
#   pass through untouched. Zygote already unthunks; delete `_mat` or guard on `AbstractArray{<:Number}`.
# - **P4** Pack-then-unpack of `[t_emb; x_t]` (`build_model_input` → `AGNN` slices it apart): two
#   allocations per forward and the awkward `in_dim = 33` convention. Pass `(g, x_t, t_emb)`.
# - **P5** Hard CUDA calls (`CUDA.rand/fill/zeros`) in `bernoulli_diffusion.jl`,
#   `discrete_flow_matching.jl`, `AGNN.jl:144`, `diffusion_model.jl:261`: nothing runs on CPU, so no
#   model/method unit test can run without a GPU and `Flux.gpu` fallback is broken. Use `similar`/`rand!`.
# - **P6** DFM differentiates `κ.(t)`/`κ̇.(t)` broadcasts of a boxed `Function` field inside
#   `withgradient` though they are constants; hoist `velocity_scale`. Second forward pass is paid even
#   for `λ_cons == 0`. `Function`-typed fields make `DiscreteFlowMatching`/`ATSSScheduler` type-unstable.
# - **P7** Training timings have no `CUDA.synchronize()` (`diffusion_model.jl:125-139`): backward and
#   optimizer work leak into the next iteration's "GPU Wait time (for solver)", so both logged numbers
#   are unreliable. DFM `sample_heatmap` logs `@info` with the full step vector on every call (`:195`;
#   Bernoulli uses `@debug`) inside the timed region.
# - **P8** `_validation_loss` runs a discarded backward pass (2 forwards + backward for DFM); validation
#   instances are solved serially on the main thread with one env idle (`test_utils.jl:80-85`).
# - **P9** ATSS fit: `lookup_x` does `argmin(abs.(fine_grid .- t))` (allocating linear scan) for every
#   cost evaluation although t is a grid point; ~10⁹ ops for S = 1000, T = 50, plus a Gurobi solve at
#   scheduler construction. `BasicGraphGenerator` materializes and shuffles all O(n²) non-tree pairs per
#   graph. Each label rebuilds a JuMP model whose build time is outside the time limit.
# - **P10** `workers = 12` vs `NUM_GUROBI_LICENSES = 2`: ten tasks permanently blocked on `take!`.