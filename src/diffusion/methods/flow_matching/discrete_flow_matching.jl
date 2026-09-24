# src/diffusion/methods/flow_matching/discrete_flow_matching.jl
#
# Implements the Discrete Flow Matching framework from the EFLOCO paper (mixture path, Eq. 2–7) with the
# consistency regularization (Eq. 9–11) structured as in the EFLOCO reference implementation
# (pl_tsp_model_consin.py), made differentiable.

using Flux
using Statistics
using Random
using Functors

export DiscreteFlowMatching

# Consistency-loss constants, taken from the EFLOCO reference implementation.
const CONS_NUM_SEGMENTS      = 19       # segment grid linspace(0, 1, 20) → 19 intervals
const CONS_DELTA             = 1f-3     # offset between the two starting points (Δ)
const CONS_ANCHOR_THRESHOLD  = 0.95f0   # if t + Δ ≥ threshold the target is the ground-truth path sample

"""
    DiscreteFlowMatching{S<:TimeSchedule{Float32}, K, KD} <: Diffusion

Represents the Discrete Flow Matching framework for combinatorial optimization.

Fields:
    - 'scheduler': The inference time schedule used.
    - 'λ_cons': Weight of the consistency loss; setting it to 0 disables it.
    - 'κ': Interpolation scheduler for the mixture path.
    - 'κ̇': Derivative of the interpolation scheduler.
    - 'sampler': Generative step update rule (:euler or :exact).
"""
struct DiscreteFlowMatching{S<:TimeSchedule{Float32}, K, KD} <: Diffusion
    scheduler::S
    λ_cons::Float32
    κ::K
    κ̇::KD
    sampler::Symbol

    # Single (inner) constructor: accepts any schedule, e.g. `LinearScheduler(1000, 50)` with Int steps,
    # and converts it to Float32. Being the only constructor, it cannot be ambiguous with a default one.
    function DiscreteFlowMatching(s::TimeSchedule, λ_cons::Real, κ::K, κ̇::KD;
                                  sampler::Symbol=:euler) where {K, KD}
        sampler in (:euler, :exact) || throw(ArgumentError("sampler must be :euler or :exact, got :$sampler"))
        s_float = convert(TimeSchedule{Float32}, s)
        return new{typeof(s_float), K, KD}(s_float, Float32(λ_cons), κ, κ̇, sampler)
    end
end

Functors.@leaf DiscreteFlowMatching

_κ(d::DiscreteFlowMatching, τ) = Float32.(d.κ.(τ))

"""
    _velocity_scale(d::DiscreteFlowMatching, τ)

Calculates the clamped probability velocity scale based on the interpolation scheduler and its derivative.

Parameters:
    - 'd': The Discrete Flow Matching configuration.
    - 'τ': The continuous time parameter.
"""
_velocity_scale(d::DiscreteFlowMatching, τ) =
    clamp.(Float32.(d.κ̇.(τ)) ./ max.(1f0 .- _κ(d, τ), 1f-6), -1f2, 1f2)

"""
    uₜ(diffusion::DiscreteFlowMatching, x_t::AbstractArray, p₁ₜ::AbstractArray, t)

Computes the probability velocity for the given state and time.

Parameters:
    - 'diffusion': The Discrete Flow Matching configuration.
    - 'x_t': The state at the current timestep.
    - 'p₁ₜ': The probability of the state being 1.
    - 't': The current continuous time parameter.
"""
uₜ(diffusion::DiscreteFlowMatching, x_t::AbstractArray, p₁ₜ::AbstractArray, t) =
    _velocity_scale(diffusion, t) .* (p₁ₜ .- x_t)

"""
    _euler_increment(x::AbstractArray, u::AbstractArray, h)

Performs an Euler step for the probability of state 1, clamped between 0 and 1.

Parameters:
    - 'x': The current probability state.
    - 'u': The probability velocity.
    - 'h': The step size.
"""
_euler_increment(x::AbstractArray, u::AbstractArray, h) = clamp.(x .+ h .* u, 0f0, 1f0)

"""
    _step_coef(d::DiscreteFlowMatching, τ, τ_next)

Calculates the coefficient for one generative step from the current to the next time.

Parameters:
    - 'd': The Discrete Flow Matching configuration.
    - 'τ': The current continuous time.
    - 'τ_next': The next continuous time.
"""
function _step_coef(d::DiscreteFlowMatching, τ, τ_next)
    if d.sampler === :exact
        κ_τ = _κ(d, τ)
        return clamp.((_κ(d, τ_next) .- κ_τ) ./ max.(1f0 .- κ_τ, 1f-6), 0f0, 1f0)
    end
    return Float32.(τ_next .- τ) .* _velocity_scale(d, τ)
end

"""
    _step(x::AbstractArray, p1::AbstractArray, c)

Applies one generative step for the probability of state 1 using the provided coefficient.

Parameters:
    - 'x': The current probability state.
    - 'p1': The probability of being 1.
    - 'c': The update coefficient.
"""
_step(x::AbstractArray, p1::AbstractArray, c) = clamp.(x .+ c .* (p1 .- x), 0f0, 1f0)

"""
    euler_step(x_t::AbstractArray, u_t::AbstractArray, τ::Real, τ_next::Real)

Integrates the state forward using an Euler step from the current to the next time.

Parameters:
    - 'x_t': The current state array.
    - 'u_t': The probability velocity array.
    - 'τ': The current time.
    - 'τ_next': The next time.
"""
euler_step(x_t::AbstractArray, u_t::AbstractArray, τ::Real, τ_next::Real) =
    _euler_increment(x_t, u_t, Float32(τ_next - τ))

"""
    _path_sample(κ_τ, x_1::AbstractArray, x_src::AbstractArray, u::AbstractArray)

Generates a sample of the mixture path using shared randomness for coupled states.

Parameters:
    - 'κ_τ': The interpolation scheduler value at time τ.
    - 'x_1': The target state array.
    - 'x_src': The source state array.
    - 'u': The shared uniform randomness array.
"""
_path_sample(κ_τ, x_1::AbstractArray, x_src::AbstractArray, u::AbstractArray) = ifelse.(u .< κ_τ, x_1, x_src)

"""
    corrupt_batch(diffusion::DiscreteFlowMatching, x_1::AbstractArray, g::GNNGraph)

Samples a random timestep for each graph and applies corruption on the mixture path.

Parameters:
    - 'diffusion': The Discrete Flow Matching configuration.
    - 'x_1': The target state array.
    - 'g': The graph structure containing batch information.
"""
function corrupt_batch(diffusion::DiscreteFlowMatching, x_1::AbstractArray, g::GNNGraph)
    T = diffusion.scheduler.total_steps
    ng = g.num_graphs
    N = size(x_1, 2)

    t_cpu = rand(Int32(0):Int32(T - 1), ng)
    t_device = copyto!(similar(x_1, Int32, ng), t_cpu)

    τ_nodes = _node_row(Float32.(t_device) ./ Float32(T), g.graph_indicator, N)

    u = rand!(similar(x_1, Float32))
    x_src = Float32.(rand!(similar(x_1, Float32)) .> 0.5f0)   # source x_0 ~ Unif{0,1}

    x_t = _path_sample(_κ(diffusion, τ_nodes), x_1, x_src, u)

    return x_t, t_device, x_1, (; x_src, u, t_cpu)
end

"""
    _segment_horizon(τ::Real)

Determines the right end of the consistency segment containing the given time on the grid.

Parameters:
    - 'τ': The current continuous time.
"""
_segment_horizon(τ::Real) =
    Float32(min(floor(Int, τ * CONS_NUM_SEGMENTS) + 1, CONS_NUM_SEGMENTS) / CONS_NUM_SEGMENTS)

"""
    _consistency_times(t_cpu::AbstractVector{<:Integer}, T::Int)

Calculates the second starting time and the common horizon for the consistency loss per graph.

Parameters:
    - 't_cpu': A vector of starting timesteps.
    - 'T': The total number of steps.
"""
function _consistency_times(t_cpu::AbstractVector{<:Integer}, T::Int)
    S = CONS_NUM_SEGMENTS
    Δ_steps = max(1, round(Int, CONS_DELTA * T))
    t_next  = similar(t_cpu, Int32)
    horizon = similar(t_cpu, Float32)
    for i in eachindex(t_cpu)
        t = Int(t_cpu[i])
        k = min(div(t * S, T) + 1, S)            # segment index (exact integer version of _segment_horizon)
        h_max = max(cld(k * T, S) - 1, t)         # last integer step strictly before the horizon k/S
        t_next[i]  = clamp(t + Δ_steps, t, h_max)
        horizon[i] = Float32(k / S)
    end
    return t_next, horizon
end

"""
    _prepare_loss(gnn::GNN, method::DiscreteFlowMatching, x_t, timesteps, batch::GNNGraph, x_1, aux, symmetric::Bool)

Prepares the constant components for the discrete flow matching and consistency loss calculations.

Parameters:
    - 'gnn': The graph neural network model.
    - 'method': The Discrete Flow Matching configuration.
    - 'x_t': The state at the current timestep.
    - 'timesteps': The timesteps for each graph in the batch.
    - 'batch': The batch of graphs.
    - 'x_1': The target state array.
    - 'aux': Auxiliary data containing shared randomness.
    - 'symmetric': Boolean flag indicating if the loss should be flip-invariant.
"""
function _prepare_loss(gnn::GNN, method::DiscreteFlowMatching, x_t, timesteps, batch::GNNGraph,
                       x_1, aux, symmetric::Bool)
    T  = method.scheduler.total_steps
    gi = batch.graph_indicator
    ng = batch.num_graphs
    N  = size(x_1, 2)

    base = (; batch, x_t, x_1, t_emb = _time_embedding(gnn, timesteps, batch), gi, ng, symmetric)
    method.λ_cons == 0 && return merge(base, (; cons = nothing))   # P6: no second forward pass

    t_next_cpu, horizon_cpu = _consistency_times(aux.t_cpu, T)
    t_next    = copyto!(similar(timesteps, Int32, ng), t_next_cpu)
    horizon_g = copyto!(similar(x_1, Float32, ng), horizon_cpu)

    τ_n  = _node_row(Float32.(timesteps) ./ Float32(T), gi, N)
    τn_n = _node_row(Float32.(t_next) ./ Float32(T), gi, N)
    hz_n = _node_row(horizon_g, gi, N)

    # coupled states on the trajectory of x_t (same x_src and u)
    x_next   = _path_sample(_κ(method, τn_n), x_1, aux.x_src, aux.u)
    x_anchor = _path_sample(_κ(method, hz_n), x_1, aux.x_src, aux.u)

    c_t = _step_coef(method, τ_n, hz_n)
    c_n = _step_coef(method, τn_n, hz_n)

    # stop-gradient target: computed outside the differentiated closure
    p_next = Flux.sigmoid.(gnn(batch, x_next, _time_embedding(gnn, t_next, batch)))
    π_next = _step(x_next, p_next, c_n)
    target = ifelse.(τn_n .>= CONS_ANCHOR_THRESHOLD, x_anchor, π_next)

    return merge(base, (; cons = (; c_t, target)))
end

"""
    L_dfm(logits::AbstractArray, x_1::AbstractArray, gi, num_graphs::Int, symmetric::Bool)

Calculates the discrete flow matching objective using cross-entropy, with optional flip-invariance.

Parameters:
    - 'logits': The predicted model outputs.
    - 'x_1': The target state array.
    - 'gi': The graph indicator mapping nodes to graphs.
    - 'num_graphs': The total number of graphs in the batch.
    - 'symmetric': Boolean flag for flip-invariance.
"""
L_dfm(logits::AbstractArray, x_1::AbstractArray, gi, num_graphs::Int, symmetric::Bool) =
    _node_bce(logits, x_1, gi, num_graphs, symmetric)

"""
    L_cons(π_t::AbstractArray, target::AbstractArray)

Calculates the consistency loss as the mean squared difference between predicted and target states.

Parameters:
    - 'π_t': The induced state at the common horizon.
    - 'target': The target state array.
"""
L_cons(π_t::AbstractArray, target::AbstractArray) = Flux.mse(π_t, target)

"""
    L_total(l_dfm, l_cons, λ_cons::Float32)

Computes the total objective by combining the discrete flow matching and consistency losses.

Parameters:
    - 'l_dfm': The discrete flow matching loss.
    - 'l_cons': The consistency loss.
    - 'λ_cons': The consistency loss weight.
"""
L_total(l_dfm, l_cons, λ_cons::Float32) = l_dfm + λ_cons * l_cons

"""
    _loss_terms(gnn::GNN, method::DiscreteFlowMatching, prep)

Calculates the differentiated parts of the loss, returning the total, DFM, and consistency losses.

Parameters:
    - 'gnn': The graph neural network model.
    - 'method': The Discrete Flow Matching configuration.
    - 'prep': The prepared loss components.
"""
function _loss_terms(gnn::GNN, method::DiscreteFlowMatching, prep)
    logits = gnn(prep.batch, prep.x_t, prep.t_emb)
    l_dfm = L_dfm(logits, prep.x_1, prep.gi, prep.ng, prep.symmetric)
    if isnothing(prep.cons)
        return (loss = l_dfm, l_dfm = l_dfm, l_cons = 0f0)
    end
    c = prep.cons
    π_t = _step(prep.x_t, Flux.sigmoid.(logits), c.c_t)
    l_cons = L_cons(π_t, c.target)
    return (loss = L_total(l_dfm, l_cons, method.λ_cons), l_dfm = l_dfm, l_cons = l_cons)
end

"""
    p(diffusion::DiscreteFlowMatching, model, x_τ::AbstractArray, τ::Real, τ_next::Real, g::GNNGraph)

Performs a single generative step, predicting the probability of state 1 and updating using the specific sampler.

Parameters:
    - 'diffusion': The Discrete Flow Matching configuration.
    - 'model': The neural network model used for prediction.
    - 'x_τ': The current state array.
    - 'τ': The current time.
    - 'τ_next': The next time.
    - 'g': The graph structure containing batch information.
"""
function p(diffusion::DiscreteFlowMatching, model, x_τ::AbstractArray, τ::Real, τ_next::Real, g::GNNGraph)
    T = diffusion.scheduler.total_steps

    t_idx = clamp(round(Int, τ * T), 0, T - 1)   # stay inside the trained range 0..T−1
    t_batch = fill!(similar(x_τ, Int32, g.num_graphs), Int32(t_idx))

    logits = model(x_τ, t_batch, g)
    p1t_pred = Flux.sigmoid.(logits)

    noise = rand!(similar(x_τ, Float32))
    if τ_next >= 1f0 - 1f-6
        return ifelse.(noise .< p1t_pred, 1.0f0, 0.0f0), p1t_pred
    end

    x_next_prob = _step(x_τ, p1t_pred, _step_coef(diffusion, Float32(τ), Float32(τ_next)))   # Eq. 6–7 / exact

    return ifelse.(noise .< x_next_prob, 1.0f0, 0.0f0), p1t_pred
end

"""
    sample_heatmap(diffusion::DiscreteFlowMatching, model, shape::Tuple, g::GNNGraph)

Executes the full forward generative process to generate a heatmap prediction of the final state.

Parameters:
    - 'diffusion': The Discrete Flow Matching configuration.
    - 'model': The neural network model used for prediction.
    - 'shape': The dimensions of the target generation.
    - 'g': The graph structure containing batch information.
"""
function sample_heatmap(diffusion::DiscreteFlowMatching, model, shape::Tuple, g::GNNGraph; kwargs...)
    x = _random_binary(g, shape)

    τ_current = diffusion.scheduler.steps[1]   # R12: read from the schedule

    @debug "Scheduler State" (; (f => getfield(diffusion.scheduler, f) for f in fieldnames(typeof(diffusion.scheduler)))...)

    p1t_pred = nothing
    for τ_step in Iterators.drop(diffusion.scheduler, 1)
        x, p1t_pred = p(diffusion, model, x, τ_current, τ_step, g)
        τ_current = τ_step
    end

    return p1t_pred
end