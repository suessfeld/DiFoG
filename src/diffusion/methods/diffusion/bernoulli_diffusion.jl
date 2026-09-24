# src/diffusion/methods/diffusion/bernoulli_diffusion.jl
#
# Implements discrete diffusion with Bernoulli noise as described in the DIFUSCO paper (§3.2, Eq. 5–6).
# Handles the forward corruption and backward denoising steps for combinatorial optimization.

using Flux
using Statistics
using Random
using NNlib
using Functors

export BernoulliDiffusion

"""
    get_betas(T::Int)

Generates a scaled linear schedule of forward-process corruption ratios for the transition matrices. The endpoints are scaled by 1000/T to ensure the final state approximates a uniform distribution.

Parameters:
    - 'T': The total number of forward steps (must be > 40).
"""
function get_betas(T::Int)
    @assert T > 40 "get_betas: total_steps must be > 40 (β_T = 20/T must stay < 0.5), got $T"
    scale = 1000f0 / Float32(T)
    return Float32.(range(1f-4 * scale, 2f-2 * scale, length=T))
end

"""
    get_alphas(T::Int)

Calculates the cumulative signal factors for the symmetric binary channel based on the forward-process corruption ratios.

Parameters:
    - 'T': The total number of forward steps.
"""
get_alphas(T::Int) = cumprod(1.0f0 .- 2.0f0 .* get_betas(T))
get_alphas(T::Int) = cumprod(1.0f0 .- 2.0f0 .* get_betas(T))

"""
    BernoulliDiffusion{S<:TimeSchedule{Int}} <: Diffusion

Represents a discrete diffusion process using Bernoulli noise. 

Fields:
    - 'scheduler': The inference time schedule used.
    - 'ᾱ': CPU-based cumulative signal factors.
    - 'ᾱ_cache': Device copy of the cumulative signal factors, cached on first use.
"""
struct BernoulliDiffusion{S<:TimeSchedule{Int}} <: Diffusion
    scheduler::S
    ᾱ::Vector{Float32}
    ᾱ_cache::Base.RefValue{Any}

    # Accepts any schedule (Float32 schedules, e.g. ATSS, are converted to Int steps).
    function BernoulliDiffusion(s::TimeSchedule)
        if s isa ATSSScheduler
            @warn "BernoulliDiffusion with an ATSS schedule: ATSS is fitted on the flow-matching κ path, not on the Bernoulli forward process (ᾱ_t); results are a transferred grid, not an adaptive schedule for this method."
        end
        s_int = convert(TimeSchedule{Int}, s)
        return new{typeof(s_int)}(s_int, get_alphas(s_int.total_steps), Ref{Any}(nothing))
    end
end

Functors.@leaf BernoulliDiffusion

"""
    _alphas_like(d::BernoulliDiffusion, x::AbstractArray)

Retrieves the cumulative signal factors on the device corresponding to the input array, utilizing a cached copy for device arrays to minimize data transfer overhead.

Parameters:
    - 'd': The Bernoulli diffusion process configuration.
    - 'x': The input array used to determine the target hardware device.
"""
function _alphas_like(d::BernoulliDiffusion, x::AbstractArray)
    x isa Array && return d.ᾱ
    c = d.ᾱ_cache[]
    if c === nothing
        c = copyto!(similar(x, Float32, length(d.ᾱ)), d.ᾱ)
        d.ᾱ_cache[] = c
    end
    return c
end

"""
    q(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{<:Integer})

Applies forward corruption to the initial state, where each entry is retained with a probability based on the cumulative signal factor, or flipped otherwise.

Parameters:
    - 'diffusion': The Bernoulli diffusion process configuration.
    - 'x_0': The initial state array.
    - 't': A vector of timesteps, containing one timestep per column of the initial state.
"""
function q(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{<:Integer})
    αs = _alphas_like(diffusion, x_0)
    ᾱₜ = αs[t]

    target_shape = ndims(x_0) == 2 ? (1, length(t)) : (1, 1, length(t))
    prob_keep = 0.5f0 .* (1.0f0 .+ reshape(ᾱₜ, target_shape))

    noise = rand!(similar(x_0, Float32))
    return ifelse.(noise .> prob_keep, 1.0f0 .- x_0, x_0)
end

"""
    corrupt_batch(diffusion::BernoulliDiffusion, x_0::AbstractArray, g::GNNGraph)

Samples a random timestep for each graph and applies forward corruption to the initial state.

Parameters:
    - 'diffusion': The Bernoulli diffusion process configuration.
    - 'x_0': The initial state array to be corrupted.
    - 'g': The graph structure containing batch information.
"""
function corrupt_batch(diffusion::BernoulliDiffusion, x_0::AbstractArray, g::GNNGraph)
    T = diffusion.scheduler.total_steps
    ng = g.num_graphs

    t_cpu = rand(Int32(1):Int32(T), ng)
    t_device = copyto!(similar(x_0, Int32, ng), t_cpu)
    t_nodes = _nodewise(t_device, g.graph_indicator, size(x_0, 2))

    return q(diffusion, x_0, t_nodes), t_device, x_0, nothing
end

"""
    _prepare_loss(gnn::GNN, ::BernoulliDiffusion, x_t, timesteps, batch::GNNGraph, x_0, aux, symmetric::Bool)

Prepares the constant components for the Bernoulli loss calculation, such as the timestep embedding.

Parameters:
    - 'gnn': The graph neural network model.
    - 'x_t': The corrupted state at the current timestep.
    - 'timesteps': The timesteps for each graph in the batch.
    - 'batch': The batch of graphs.
    - 'x_0': The initial state array.
    - 'aux': Auxiliary data.
    - 'symmetric': Boolean flag indicating if the loss should be flip-invariant.
"""
_prepare_loss(gnn::GNN, ::BernoulliDiffusion, x_t, timesteps, batch::GNNGraph, x_0, aux, symmetric::Bool) =
    (; batch, x_t, x_0, t_emb = _time_embedding(gnn, timesteps, batch),
       gi = batch.graph_indicator, ng = batch.num_graphs, symmetric)

"""
    _loss_terms(gnn::GNN, ::BernoulliDiffusion, prep)

Calculates the cross-entropy loss between the model's prediction and the initial state, with optional flip-invariance.

Parameters:
    - 'gnn': The graph neural network model.
    - 'prep': The prepared loss components containing batch, states, and embeddings.
"""
function _loss_terms(gnn::GNN, ::BernoulliDiffusion, prep)
    logits = gnn(prep.batch, prep.x_t, prep.t_emb)
    return (loss = _node_bce(logits, prep.x_0, prep.gi, prep.ng, prep.symmetric),)
end

"""
    posterior_prob(diffusion::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int, s::Int)

Calculates the posterior probability of the state at a previous timestep given the current state and the model's prediction of the initial state.

Parameters:
    - 'diffusion': The Bernoulli diffusion process configuration.
    - 'x_t': The state at the current timestep.
    - 'x_0_pred': The model's prediction of the initial state.
    - 't': The current timestep.
    - 's': The target previous timestep.
"""
function posterior_prob(diffusion::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int, s::Int)
    ᾱₜ = diffusion.ᾱ[t]                              # CPU scalars (no 1-element device arrays)
    ᾱₛ = s > 0 ? diffusion.ᾱ[s] : 1.0f0

    α_t_s = ᾱₜ / ᾱₛ
    p_t_s = 0.5f0 * (1.0f0 + α_t_s)

    # q(x_t | x_s = 1)
    term_1 = x_t .* p_t_s .+ (1.0f0 .- x_t) .* (1.0f0 - p_t_s)

    # Eq. 5 for x_0 = 1: q(x_s=1 | x_t, x_0=1) = q(x_t | x_s=1) q(x_s=1 | x_0=1) / q(x_t | x_0=1)
    term_2 = (term_1 .* (0.5f0 * (1.0f0 + ᾱₛ))) ./
             max.(x_t .* (0.5f0 * (1.0f0 + ᾱₜ)) .+ (1.0f0 .- x_t) .* (0.5f0 * (1.0f0 - ᾱₜ)), 1f-6)

    # Eq. 5 for x_0 = 0: q(x_s=1 | x_t, x_0=0) = q(x_t | x_s=1) q(x_s=1 | x_0=0) / q(x_t | x_0=0)
    term_3 = (term_1 .* (0.5f0 * (1.0f0 - ᾱₛ))) ./
             max.(x_t .* (0.5f0 * (1.0f0 - ᾱₜ)) .+ (1.0f0 .- x_t) .* (0.5f0 * (1.0f0 + ᾱₜ)), 1f-6)

    # Eq. 6: p_θ(x_s = 1 | x_t)
    term_4 = x_0_pred .* term_2 .+ (1.0f0 .- x_0_pred) .* term_3

    return clamp.(term_4, 0.0f0, 1.0f0)
end

"""
    p(diffusion::BernoulliDiffusion, model, x_t::AbstractArray, t::Int, s::Int, g::GNNGraph)

Performs a single reverse denoising step from the current timestep to a previous timestep.

Parameters:
    - 'diffusion': The Bernoulli diffusion process configuration.
    - 'model': The neural network model used for prediction.
    - 'x_t': The corrupted state at the current timestep.
    - 't': The current timestep.
    - 's': The target previous timestep.
    - 'g': The graph structure containing batch information.
"""
function p(diffusion::BernoulliDiffusion, model, x_t::AbstractArray, t::Int, s::Int, g::GNNGraph)
    t_batch = fill!(similar(x_t, Int32, g.num_graphs), Int32(t))

    logits = model(x_t, t_batch, g)
    x_0_pred = Flux.sigmoid.(logits)
    posterior = posterior_prob(diffusion, x_t, x_0_pred, t, s)

    noise = rand!(similar(x_t, Float32))
    x_s = ifelse.(noise .< posterior, 1.0f0, 0.0f0)

    return x_s, x_0_pred
end

"""
    sample_heatmap(diffusion::BernoulliDiffusion, model, shape::Tuple, g::GNNGraph; kwargs...)

Executes the full reverse diffusion inference process to generate a heatmap prediction of the initial state.

Parameters:
    - 'diffusion': The Bernoulli diffusion process configuration.
    - 'model': The neural network model used for prediction.
    - 'shape': The dimensions of the target generation.
    - 'g': The graph structure containing batch information.
"""
function sample_heatmap(diffusion::BernoulliDiffusion, model, shape::Tuple, g::GNNGraph; kwargs...)
    x = _random_binary(g, shape)

    τ_current = diffusion.scheduler.steps[end]

    @debug "Scheduler State" (; (f => getfield(diffusion.scheduler, f) for f in fieldnames(typeof(diffusion.scheduler)))...)

    x_0_pred = nothing
    for τ_prev in Iterators.drop(Iterators.reverse(diffusion.scheduler), 1)
        x, x_0_pred = p(diffusion, model, x, τ_current, τ_prev, g)
        τ_current = τ_prev
    end

    return x_0_pred
end