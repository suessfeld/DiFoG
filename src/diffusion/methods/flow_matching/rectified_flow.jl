# src/diffusion/methods/flow_matching/rectified_flow.jl
#
# Implements Rectified Flow and the Reflow procedure (Liu et al., "Flow Straight and Fast", 2022) for
# combinatorial optimization via a continuous relaxation of the binary node labels.
#
# Rectified Flow lives in the continuous domain, so the binary labels are relaxed:
#   - labels {0, 1} are mapped to the signal space {−1, +1} (x₁ = 2y − 1), as in DIFUSCO's Gaussian variant,
#   - the source is x₀ ~ N(0, σ²I),
#   - the path is the straight line x_τ = τ x₁ + (1 − τ) x₀ with constant velocity x₁ − x₀,
#   - generation integrates the ODE dx/dτ = v_θ(x_τ, τ) from τ = 0 to τ = 1 (deterministic given x₀).
#
# Reflow: a trained model defines a deterministic coupling (x₀, Z₁ = ODE(x₀)). Retraining on this
# coupling instead of the independent one (x₀, x₁) straightens the flow, so that few (ideally one) Euler
# steps suffice. The coupling is stored in the graphs' `ndata` (`x` = Z₁ in label space, `x0` = x₀), so the
# regular `train_diffusion_model!` loop is reused unchanged: `corrupt_batch` takes x₀ from `ndata.x0` when
# present and samples it otherwise.

using Flux
using Statistics
using Random
using NNlib
using Functors
using GraphNeuralNetworks

export RectifiedFlow, generate_reflow_coupling, straightness

"""
    RectifiedFlow{S<:TimeSchedule{Float32}} <: Diffusion

Represents a (continuous) Rectified Flow on the relaxed node labels.

Fields:
    - 'scheduler': The inference time schedule used (Euler integration grid).
    - 'prediction': Network parameterization, `:x1` (endpoint logit, BCE loss) or `:velocity` (velocity, MSE loss).
    - 'σ_src': Standard deviation of the Gaussian source distribution.
"""
struct RectifiedFlow{S<:TimeSchedule{Float32}} <: Diffusion
    scheduler::S
    prediction::Symbol
    σ_src::Float32

    function RectifiedFlow(s::TimeSchedule; prediction::Symbol=:x1, σ_src::Real=1f0)
        prediction in (:x1, :velocity) ||
            throw(ArgumentError("prediction must be :x1 or :velocity, got :$prediction"))
        σ_src > 0 || throw(ArgumentError("σ_src must be > 0, got $σ_src"))
        if s isa ATSSScheduler && s.path !== :rectified_flow
            @warn "RectifiedFlow with an ATSS schedule fitted on the $(s.path) path, not on the straight Gaussian path (fit it with path=:rectified_flow); results are a transferred grid, not an adaptive schedule for this method."
        end
        s_float = convert(TimeSchedule{Float32}, s)
        return new{typeof(s_float)}(s_float, prediction, Float32(σ_src))
    end
end

Functors.@leaf RectifiedFlow

"""
    _to_signal(y)

Maps labels in [0, 1] to the signal space [−1, 1].
"""
_to_signal(y) = 2f0 .* y .- 1f0

"""
    _to_label(x)

Maps the signal space [−1, 1] back to labels in [0, 1].
"""
_to_label(x) = (x .+ 1f0) ./ 2f0

"""
    _random_gaussian(g::GNNGraph, shape::Tuple, σ::Float32)

Samples a Gaussian source state directly on the same device as the input graph.

Parameters:
    - 'g': The reference graph determining the target device.
    - 'shape': The dimensions of the generated state.
    - 'σ': The standard deviation.
"""
_random_gaussian(g::GNNGraph, shape::Tuple, σ::Float32) =
    σ .* randn!(similar(_device_anchor(g), Float32, shape))

"""
    _has_coupling(g::GNNGraph)

Checks whether the graph carries a stored Reflow source state (`ndata.x0`).
"""
_has_coupling(g::GNNGraph) = :x0 in keys(g.ndata)

"""
    _interpolate(τ, x_1::AbstractArray, x_0::AbstractArray)

Evaluates the straight-line path x_τ = τ x₁ + (1 − τ) x₀.

Parameters:
    - 'τ': The time (scalar or row broadcastable against the states).
    - 'x_1': The target state in signal space.
    - 'x_0': The source state.
"""
_interpolate(τ, x_1::AbstractArray, x_0::AbstractArray) = τ .* x_1 .+ (1f0 .- τ) .* x_0

"""
    corrupt_batch(flow::RectifiedFlow, y::AbstractArray, g::GNNGraph)

Samples a random time for each graph and places the state on the straight path between the source and
the (relaxed) target. Uses the stored Reflow source `ndata.x0` if present (fixed coupling), otherwise
samples an independent Gaussian source.

Parameters:
    - 'flow': The Rectified Flow configuration.
    - 'y': The target labels in [0, 1] (binary for 1-RF, soft for Reflow couplings).
    - 'g': The graph structure containing batch information.
"""
function corrupt_batch(flow::RectifiedFlow, y::AbstractArray, g::GNNGraph)
    T = flow.scheduler.total_steps
    ng = g.num_graphs
    N = size(y, 2)

    t_cpu = rand(Int32(0):Int32(T - 1), ng)
    t_device = copyto!(similar(y, Int32, ng), t_cpu)
    τ_nodes = _node_row(Float32.(t_device) ./ Float32(T), g.graph_indicator, N)

    coupled = _has_coupling(g)
    x_0 = coupled ? g.ndata.x0 : _random_gaussian(g, size(y), flow.σ_src)
    x_1 = _to_signal(y)

    x_t = _interpolate(τ_nodes, x_1, x_0)

    return x_t, t_device, y, (; x_0, coupled)
end

"""
    _prepare_loss(gnn::GNN, flow::RectifiedFlow, x_t, timesteps, batch::GNNGraph, y, aux, symmetric::Bool)

Prepares the constant components of the Rectified Flow loss. The flip-invariant loss is disabled for a
stored Reflow coupling: Z₁ is paired with a fixed x₀, so its complement is not an equally valid target.

Parameters:
    - 'gnn': The graph neural network model.
    - 'flow': The Rectified Flow configuration.
    - 'x_t': The state on the path.
    - 'timesteps': The timesteps for each graph in the batch.
    - 'batch': The batch of graphs.
    - 'y': The target labels in [0, 1].
    - 'aux': Auxiliary data containing the source state and the coupling flag.
    - 'symmetric': Boolean flag indicating if the loss should be flip-invariant.
"""
_prepare_loss(gnn::GNN, flow::RectifiedFlow, x_t, timesteps, batch::GNNGraph, y, aux, symmetric::Bool) =
    (; batch, x_t, y, x_0 = aux.x_0, t_emb = _time_embedding(gnn, timesteps, batch),
       τ = _node_row(Float32.(timesteps) ./ Float32(flow.scheduler.total_steps), batch.graph_indicator, size(y, 2)),
       gi = batch.graph_indicator, ng = batch.num_graphs, symmetric = symmetric && !aux.coupled)

"""
    _velocity_targets(prep)

Returns the `:velocity` regression targets `(v, v_flip)`: v = x₁ − x₀ (equal to (x₁ − x_τ)/(1 − τ)) and the
velocity (−x₁ − x_τ)/(1 − τ) that reaches the complement −x₁ from the same x_τ. The flipped target grows like
(1 + τ)/(1 − τ), so the per-graph minimum picks it only for small τ, where x_τ carries almost no information
about the orientation (τ ≤ 1 − 1/T, since training times are drawn from 0:T−1).

Parameters:
    - 'prep': The prepared loss components.
"""
function _velocity_targets(prep)
    x_1 = _to_signal(prep.y)
    return x_1 .- prep.x_0, (.-x_1 .- prep.x_t) ./ (1f0 .- prep.τ)
end

"""
    _node_mse(pred::AbstractArray, y::AbstractArray, y_flip::AbstractArray, gi, num_graphs::Int, symmetric::Bool)

Calculates the mean squared error over all nodes, optionally taking per graph the minimum over the
target and its flipped counterpart.

Parameters:
    - 'pred': The predicted values.
    - 'y': The regression target.
    - 'y_flip': The regression target for the complemented labels.
    - 'gi': The graph indicator mapping nodes to graphs.
    - 'num_graphs': The total number of graphs in the batch.
    - 'symmetric': Boolean flag to evaluate the loss up to complement symmetry.
"""
function _node_mse(pred::AbstractArray, y::AbstractArray, y_flip::AbstractArray, gi, num_graphs::Int,
                   symmetric::Bool)
    symmetric || return Flux.mse(pred, y)
    l_a = (pred .- y) .^ 2
    l_b = (pred .- y_flip) .^ 2
    if isnothing(gi)
        return min(sum(l_a), sum(l_b)) / length(y)
    end
    s_a = NNlib.scatter(+, vec(l_a), gi; dstsize=(num_graphs,))
    s_b = NNlib.scatter(+, vec(l_b), gi; dstsize=(num_graphs,))
    return sum(min.(s_a, s_b)) / length(y)
end

"""
    _loss_terms(gnn::GNN, flow::RectifiedFlow, prep)

Calculates the Rectified Flow objective.
    - `:x1`: BCE between the endpoint logit and the (possibly soft) target label. Its minimizer is
      E[x₁ | x_τ], which induces exactly the rectified velocity (E[x₁ | x_τ] − x_τ)/(1 − τ).
    - `:velocity`: MSE between the predicted velocity and x₁ − x₀ (Liu et al., Eq. 1); the flipped target
      is the velocity from x_τ to −x₁ (see `_velocity_targets`).

Parameters:
    - 'gnn': The graph neural network model.
    - 'flow': The Rectified Flow configuration.
    - 'prep': The prepared loss components.
"""
function _loss_terms(gnn::GNN, flow::RectifiedFlow, prep)
    out = gnn(prep.batch, prep.x_t, prep.t_emb)
    if flow.prediction === :x1
        return (loss = _node_bce(out, prep.y, prep.gi, prep.ng, prep.symmetric),)
    end
    v, v_flip = _velocity_targets(prep)
    return (loss = _node_mse(out, v, v_flip, prep.gi, prep.ng, prep.symmetric),)
end

"""
    _velocity(flow::RectifiedFlow, out::AbstractArray, x::AbstractArray, τ::Float32)

Converts the network output into the ODE velocity and the endpoint estimate (signal space).

Parameters:
    - 'flow': The Rectified Flow configuration.
    - 'out': The raw network output.
    - 'x': The current state.
    - 'τ': The current time.
"""
function _velocity(flow::RectifiedFlow, out::AbstractArray, x::AbstractArray, τ::Float32)
    if flow.prediction === :x1
        x̂_1 = _to_signal(Flux.sigmoid.(out))
        return (x̂_1 .- x) ./ max(1f0 - τ, 1f-6), x̂_1
    end
    return out, x .+ (1f0 - τ) .* out
end

"""
    p(flow::RectifiedFlow, model, x_τ::AbstractArray, τ::Real, τ_next::Real, g::GNNGraph)

Performs a single deterministic Euler step of the flow ODE.

Parameters:
    - 'flow': The Rectified Flow configuration.
    - 'model': The neural network model used for prediction, called as `model(x, t, g)`.
    - 'x_τ': The current state (signal space).
    - 'τ': The current time.
    - 'τ_next': The next time.
    - 'g': The graph structure containing batch information.

Returns the next state, the endpoint estimate (signal space) and the velocity used.
"""
function p(flow::RectifiedFlow, model, x_τ::AbstractArray, τ::Real, τ_next::Real, g::GNNGraph)
    T = flow.scheduler.total_steps
    t_idx = clamp(round(Int, τ * T), 0, T - 1)   # stay inside the trained range 0..T−1
    t_batch = fill!(similar(x_τ, Int32, g.num_graphs), Int32(t_idx))

    v, x̂_1 = _velocity(flow, model(x_τ, t_batch, g), x_τ, Float32(τ))
    return x_τ .+ Float32(τ_next - τ) .* v, x̂_1, v
end

"""
    integrate(flow::RectifiedFlow, model, x_0::AbstractArray, g::GNNGraph, τs::AbstractVector{<:Real})

Integrates the flow ODE with Euler steps on the grid `τs` (ascending, from 0 to 1) starting at `x_0`.
Returns the final state in signal space.

Parameters:
    - 'flow': The Rectified Flow configuration.
    - 'model': The neural network model used for prediction.
    - 'x_0': The source state.
    - 'g': The graph structure containing batch information.
    - 'τs': The integration grid.
"""
function integrate(flow::RectifiedFlow, model, x_0::AbstractArray, g::GNNGraph, τs::AbstractVector{<:Real})
    x = x_0
    for i in 1:(length(τs) - 1)
        x, _, _ = p(flow, model, x, τs[i], τs[i + 1], g)
    end
    return x
end

"""
    _uniform_grid(n::Int)

Uniform integration grid with `n` Euler steps on [0, 1].
"""
_uniform_grid(n::Int) = Float32.(range(0, 1, length=n + 1))

"""
    sample_heatmap(flow::RectifiedFlow, model, shape::Tuple, g::GNNGraph; kwargs...)

Integrates the flow ODE from Gaussian noise on the inference schedule and returns the final state mapped
to label space and clamped to [0, 1] as the heatmap.

Parameters:
    - 'flow': The Rectified Flow configuration.
    - 'model': The neural network model used for prediction.
    - 'shape': The dimensions of the target generation.
    - 'g': The graph structure containing batch information.
"""
function sample_heatmap(flow::RectifiedFlow, model, shape::Tuple, g::GNNGraph; kwargs...)
    @debug "Scheduler State" (; (f => getfield(flow.scheduler, f) for f in fieldnames(typeof(flow.scheduler)))...)
    x_1 = integrate(flow, model, _random_gaussian(g, shape, flow.σ_src), g, flow.scheduler.steps)
    return clamp.(_to_label(x_1), 0f0, 1f0)
end

"""
    straightness(flow::RectifiedFlow, model, g::GNNGraph, shape::Tuple; ode_steps::Int=100)

Estimates the straightness S = ∫₀¹ E‖(Z₁ − Z₀) − v(Z_τ, τ)‖² dτ of the learned flow (Liu et al., Def. 3.2),
averaged per node, by integrating with `ode_steps` uniform Euler steps. S = 0 means perfectly straight
trajectories, which a single Euler step integrates exactly. Also returns the one-step endpoint error
E‖Z₁ − Ẑ₁^(1-step)‖² per node.

Parameters:
    - 'flow': The Rectified Flow configuration.
    - 'model': The neural network model used for prediction.
    - 'g': The graph structure containing batch information.
    - 'shape': The dimensions of the state.
    - 'ode_steps': The number of Euler steps of the reference trajectory.
"""
function straightness(flow::RectifiedFlow, model, g::GNNGraph, shape::Tuple; ode_steps::Int=100)
    τs = _uniform_grid(ode_steps)
    x_0 = _random_gaussian(g, shape, flow.σ_src)

    x = x_0
    vs = Vector{typeof(x_0)}(undef, ode_steps)
    for i in 1:ode_steps
        x, _, vs[i] = p(flow, model, x, τs[i], τs[i + 1], g)
    end
    Δ = x .- x_0
    S = sum(Float32(τs[i + 1] - τs[i]) * mean((Δ .- vs[i]) .^ 2) for i in 1:ode_steps)

    one_step = x_0 .+ vs[1]   # a single Euler step from τ = 0 uses exactly the first velocity
    return (; straightness = Float32(S), one_step_error = Float32(mean((x .- one_step) .^ 2)))
end

"""
    generate_reflow_coupling(model::DiffusionModel{<:GNN,<:RectifiedFlow}, data_loader; ode_steps::Int=100,
                             pairs_per_instance::Int=1)

Generates the Reflow coupling (x₀, Z₁ = ODE(x₀)) with the trained flow for every instance of the data
loader. Returns a vector of `(graph, name)` tuples whose `ndata.x` holds Z₁ in label space (soft labels)
and `ndata.x0` the source x₀; all other graph data is kept. The result can be wrapped in a
`Flux.DataLoader(...; collate=true)` and passed to `train_diffusion_model!`.

With `pairs_per_instance = K > 1`, the loader is traversed K times and every pass draws a fresh x₀ per
instance, so the coupling covers the source distribution with K pairs per graph instead of one (Liu et al.
generate a large set of pairs; with a single x₀ per graph, the 2-RF mainly learns those trajectories).
A `RuntimeDataset` supports one pass per producer: pass a fresh one or `reset!` it beforehand; for later
passes it is reset automatically.

Parameters:
    - 'model': The trained Rectified Flow model defining the coupling.
    - 'data_loader': Iterable of batched `(GNNGraph, names)` (e.g. a collating `Flux.DataLoader`).
    - 'ode_steps': The number of uniform Euler steps used to integrate the ODE (should be fine).
    - 'pairs_per_instance': The number of (x₀, Z₁) pairs generated per instance.
"""
function generate_reflow_coupling(model::DiffusionModel{<:GNN,<:RectifiedFlow}, data_loader;
                                  ode_steps::Int=100, pairs_per_instance::Int=1)
    pairs_per_instance >= 1 || throw(ArgumentError("pairs_per_instance must be ≥ 1, got $pairs_per_instance"))
    flow = model.method
    τs = _uniform_grid(ode_steps)
    coupling = Tuple{GNNGraph,String}[]

    Flux.testmode!(model.gnn)
    try
        for pass in 1:pairs_per_instance
            if pass > 1 && data_loader isa Flux.DataLoader && data_loader.data isa RuntimeDataset
                reset!(data_loader.data)
            end
            for (g_batch, names) in data_loader
                g_gpu = g_batch |> _model_device(model)
                shape = (model.gnn.config.in_dim, g_gpu.num_nodes)
                x_0 = _random_gaussian(g_gpu, shape, flow.σ_src)
                z_1 = integrate(flow, model, x_0, g_gpu, τs)

                g_new = GNNGraph(g_batch; ndata = (; g_batch.ndata..., x = Flux.cpu(_to_label(z_1)), x0 = Flux.cpu(x_0)))
                graphs = Flux.unbatch(g_new)
                @assert length(graphs) == length(names) "expected one name per graph, got $(length(graphs)) graphs and $(length(names)) names"
                append!(coupling, zip(graphs, names))
            end
        end
    finally
        Flux.trainmode!(model.gnn)
    end

    @info "Generated Reflow coupling: $(length(coupling)) pairs ($pairs_per_instance per instance, $ode_steps Euler steps)."
    return coupling
end
