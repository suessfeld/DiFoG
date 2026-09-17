# src/diffusion/methods/flow_matching/discrete_flow_matching.jl
#
# Implements the Discrete Flow Matching framework from the EFLOCO paper.
# Manages continuous time integration, velocity fields, and consistency losses for stable transition paths.

using Flux
using Statistics
using Random
using CUDA
using Functors

export DiscreteFlowMatching

"""
    DiscreteFlowMatching <: Diffusion

Struct to hold parameters for Discrete Flow Matching.

Implements the discrete flow matching framework for combinatorial optimization 
as described in the EFLOCO paper.

Fields:
    - 'scheduler': The scheduler defining the interpolation steps.
    - 'λ_cons': hyperparameter balancing the consistency loss.
    - 'κ': The interpolation function.
    - 'κ̇': The derivative of the interpolation function.
"""
struct DiscreteFlowMatching <: Diffusion
    scheduler::Scheduler{Float32} # scheduler (ATSS used in original paper)
    λ_cons::Float32 # hyperparameter that balances the consistency loss of the loss function (eq. 11)
    κ::Function # interpolation function of the scheduler
    κ̇ ::Function # derivative of κ
end

Functors.@leaf DiscreteFlowMatching

"""
    uₜ(diffusion::DiscreteFlowMatching, x_t::AbstractArray, p₁ₜ::AbstractArray, t)

Computes the transition tendency or probabilistic velocity field guiding the evolution 
of the state along the path. Corresponds to Eq. 7 in the EFLOCO paper.

Parameters:
    - 'diffusion': The flow matching method.
    - 'x_t': The current intermediate state.
    - 'p₁ₜ': The model's predicted marginal distribution for the target.
    - 't': The current timestep.
"""
function uₜ(diffusion::DiscreteFlowMatching, x_t::AbstractArray,
                         p₁ₜ::AbstractArray, t)
    κ_t  = diffusion.κ.(t)
    κ̇_t  = diffusion.κ̇.(t)
    velocity_scale = clamp.(κ̇_t ./ max.(1f0 .- κ_t, 1f-6), -1f2, 1f2)  # bound the coefficient, to avoid explosion! This is not explicitly mentioned in the paper, but ATSS pushes steps towards 1, which results in extreme values.
    return velocity_scale .* (p₁ₜ .- x_t)   # eq. 7
end

"""
    forward_transition_step(diffusion::DiscreteFlowMatching, x_t::AbstractArray, x_1::AbstractArray, t, t_next)

Determines the next state in the forward probability path between the current state and the target solution

Parameters:
    - 'diffusion': The flow matching method.
    - 'x_t': The current state.
    - 'x_1': The target ground-truth solution.
    - 't': The current timestep.
    - 't_next': The next timestep.
"""
function forward_transition_step(diffusion::DiscreteFlowMatching, x_t::AbstractArray,
                             x_1::AbstractArray, t, t_next)
    κ      = diffusion.κ.(t)
    κ_next = diffusion.κ.(t_next)
    
    p_commit = clamp.((κ_next .- κ) ./ max.(1f0 .- κ, 1f-6), 0f0, 1f0) # eq. 6 & eq. 7. 
    
    already_committed = x_t .== x_1 # eq. 4
    
    r = CUDA.rand(Float32, size(x_t))
    noise = CUDA.rand(Float32, size(x_t))
    fresh_noise = ifelse.(noise .> 0.5f0, 1.0f0, 0.0f0)

    return ifelse.(already_committed .| (r .< p_commit), x_1, fresh_noise) # eq. 6
end

"""
    corrupt_batch(diffusion::DiscreteFlowMatching, x_1::AbstractArray, g::GNNGraph)

Executes the forward transition process to generate corrupted states along the 
flow matching probability path for a batch of graphs.

Parameters:
    - 'diffusion': The flow matching method.
    - 'x_1': The target ground-truth solution.
    - 'g': The batch of GNNGraphs.
"""
function corrupt_batch(diffusion::DiscreteFlowMatching, x_1::AbstractArray, g::GNNGraph)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1 # check for batch size
    T = diffusion.scheduler.total_steps

    t_cpu = rand(Int32(1):Int32(T), batch_size)
    t_device = Flux.gpu(t_cpu)

    if hasproperty(g, :graph_indicator) && !isnothing(g.graph_indicator)
        t_nodes = t_device[g.graph_indicator]
    else
        num_nodes = size(x_1, 2)
        t_nodes = Flux.gpu(fill(Int32(t_cpu[1]), num_nodes))
    end

    τ_nodes = Float32.(t_nodes) ./ Float32(T)
    shp = _time_shape(x_1, length(τ_nodes))

    κ_reshaped = reshape(diffusion.κ.(τ_nodes), shp)

    u = CUDA.rand(Float32, size(x_1))
    mask = u .< κ_reshaped

    noise = CUDA.rand(Float32, size(x_1))
    base_noise = ifelse.(noise .> 0.5f0, 1.0f0, 0.0f0)

    corrupted_x = ifelse.(mask, x_1, base_noise)

    return corrupted_x, t_device, x_1
end

"""
    p(diffusion::DiscreteFlowMatching, model, x_τ::AbstractArray, τ::Real, τ_next::Real, g::GNNGraph)

Executes one step of the reverse generative process by computing the velocity 
field and performing an Euler integration step. 

Parameters:
    - 'diffusion': The flow matching method.
    - 'model': The neural network model predicting the target distribution.
    - 'x_τ': The current state.
    - 'τ': The current continuous time value.
    - 'τ_next': The next continuous time value.
    - 'g': The graph batch.
"""
function p(diffusion::DiscreteFlowMatching, model, x_τ::AbstractArray, τ::Real, τ_next::Real, g::GNNGraph)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    T = diffusion.scheduler.total_steps

    t_idx = clamp(round(Int, τ * T), 0, T)
    t_batch = CUDA.fill(Int32(t_idx), batch_size)

    logits = model(x_τ, t_batch, g)
    p1t_pred = Flux.sigmoid.(logits)

    if τ_next >= 1f0 - 1f-6
        noise = CUDA.rand(Float32, size(x_τ))
        return ifelse.(noise .< p1t_pred, 1.0f0, 0.0f0)
    end

    u_τ = uₜ(diffusion, x_τ, p1t_pred, Float32(τ))   # Eq. 7
    x_next_prob = euler_step(x_τ, u_τ, τ, τ_next)   # Eq. 6

    noise = CUDA.rand(Float32, size(x_τ))
    return ifelse.(noise .< x_next_prob, 1.0f0, 0.0f0)
end

"""
    euler_step(x_t::AbstractArray, u_t::AbstractArray, τ::Real, τ_next::Real)

Performs an Euler integration step to update the state probability distribution 
using the computed velocity field. Corresponds to Eq. 6 in the EFLOCO paper.

Parameters:
    - 'x_t': The current state distribution.
    - 'u_t': The computed velocity field.
    - 'τ': The current continuous time.
    - 'τ_next': The next continuous time.
"""
function euler_step(x_t::AbstractArray, u_t::AbstractArray, τ::Real, τ_next::Real)
    h = Float32(τ_next - τ)
    return clamp.(x_t .+ h .* u_t, 0f0, 1f0)
end

"""
    sample_heatmap(diffusion::DiscreteFlowMatching, model, shape::Tuple, g::GNNGraph; kwargs...)

Represents an inference call and produces a heatmap for the provided input.

Parameters:
    - 'diffusion': The flow matching method.
    - 'model': The neural network model.
    - 'shape': The shape of the required state matrix.
    - 'g': The input graph instance.
"""
function sample_heatmap(diffusion::DiscreteFlowMatching, model, shape::Tuple, g::GNNGraph; kwargs...)
    x_cpu = Float32.(rand(Float32, shape) .> 0.5f0)
    x = Flux.gpu(x_cpu)
    
    τ_current = 0.0f0

    @info "Scheduler State" (; (f => getfield(diffusion.scheduler, f) for f in fieldnames(typeof(diffusion.scheduler)))...)
    for τ_step in Iterators.drop(diffusion.scheduler, 1)
        x = p(diffusion, model, x, τ_current, τ_step, g)
        τ_current = τ_step
    end

    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = CUDA.fill(Int32(diffusion.scheduler.total_steps), batch_size)

    #GR Warum hier das Modell nochmals inferenzen, hier ist beim Training bei t = T κ(1) = 1 und damit  der Trainingsinput genau gleich dem Target, d.h. das Modell lernt die Identitätsfunktion?
    logits = model(x, t_batch, g)
    return Flux.sigmoid.(logits)
end

"""
    L_dfm(logits::AbstractArray, x_1::AbstractArray)

Calculates the discrete flow matching objective via logit binary cross-entropy. See eq. 8.
"""
function L_dfm(logits::AbstractArray, x_1::AbstractArray)
    return Flux.logitbinarycrossentropy(logits, x_1)
end

"""
    L_total(l_dfm, l_cons, λ_cons::Float32)

Calculates the combined total loss aggregating the standard flow matching loss 
and the velocity-based consistency loss. See eq. 11.
"""
function L_total(l_dfm, l_cons, λ_cons::Float32)
    return l_dfm + λ_cons * l_cons
end

"""
    L_cons(u_t::AbstractArray, u_t_next::AbstractArray)

Calculates the velocity-based consistency loss penalizing the deviation 
between the induced next-step distributions from adjacent states. See eq. 10.
"""
function L_cons(u_t::AbstractArray, u_t_next::AbstractArray)
    return Flux.mse(u_t, u_t_next)
end

"""
    _nodewise_tau(timesteps::AbstractVector, T::Int, gi, num_nodes::Int, shp)

Helper function to broadcast normalized continuous time values τ to all nodes 
within their corresponding graphs in the batch.
"""
function _nodewise_tau(timesteps::AbstractVector, T::Int, gi, num_nodes::Int, shp)
    τ = Float32.(timesteps) ./ Float32(T)
    τ_nodes = isnothing(gi) ? repeat(τ, num_nodes) : τ[gi]
    return reshape(τ_nodes, shp)
end

"""
    _time_shape(x::AbstractArray, n::Int)

Helper function determining the shape for timestep broadcasting based on the 
dimensionality of the input data matrix.
"""
_time_shape(x::AbstractArray, n::Int) = ndims(x) == 2 ? (1, n) : (1, 1, n)


"""
    Thesis.compute_loss_and_gradients(gnn::GNN, method::DiscreteFlowMatching, ...)

Executes the forward and backward passes to compute the total flow matching loss 
and gradients for the GNN model, explicitly coupling states at adjacent timesteps 
to calculate the consistency regularization.

Parameters:
    - 'gnn': The GNN network model.
    - 'method': The flow matching method.
    - 'x_t': The generated corrupted state.
    - 'timesteps': The sampled discrete timesteps.
    - 'batch': The batch of target GNNGraphs.
    - 'x_1': The target ground-truth solution.
"""
function Thesis.compute_loss_and_gradients(gnn::GNN, method::DiscreteFlowMatching,
                                            x_t::AbstractArray, timesteps::AbstractVector,
                                            batch::GNNGraph, x_1::AbstractArray)
    x_dim = size(x_t, 1)
    time_dim = gnn.config.in_dim - x_dim
    gi = batch.graph_indicator
    T = method.scheduler.total_steps
    x_input_t = Thesis.build_model_input(x_t, timesteps, time_dim, gi)

    timesteps_next = min.(T, timesteps .+ 1)
    shp = _time_shape(x_1, isnothing(gi) ? size(x_1, 2) * length(timesteps) : length(gi))

    τ_r      = _nodewise_tau(timesteps, T, gi, size(x_1, 2), shp)
    τ_next_r = _nodewise_tau(timesteps_next, T, gi, size(x_1, 2), shp)

    x_t_next = forward_transition_step(method, x_t, x_1, τ_r, τ_next_r)
    x_input_t_next = Thesis.build_model_input(x_t_next, timesteps_next, time_dim, gi)

    # Eq. 11 (total loss = l_dfm + λ_cons * L_cons)
    loss, grads = Flux.withgradient(gnn) do gnn_inner
        logits_t      = gnn_inner(batch, x_input_t)
        logits_t_next = gnn_inner(batch, x_input_t_next)

        p1t_pred      = Flux.sigmoid.(logits_t)
        p1t_next_pred = Flux.sigmoid.(logits_t_next)

        l_dfm = L_dfm(logits_t, x_1)

        u_t      = uₜ(method, x_t,      p1t_pred,      τ_r)
        u_t_next = uₜ(method, x_t_next, p1t_next_pred, τ_next_r)

        l_cons = L_cons(u_t, u_t_next)

        return L_total(l_dfm, l_cons, method.λ_cons)
    end

    return loss, (Thesis._mat(grads[1]),)
end


#GR TODO Claude liefert in Bezug auf die Kernfunktionalität folgende Meldungen, und ich denke vieles darin ist sehr begründet. Bitte dem sorgfältig nachgehen und hier dokumentieren, wie das gefixed wurde bzw. warum vielleicht manches ungerechtfertigt ist.


# - **B1** DFM consistency branch uses a biased commit proxy: `already_committed = x_t .== x_1`
#   (`discrete_flow_matching.jl:75`) although `corrupt_batch` has the exact mask (`:115`) and discards it.
#   About half the uncommitted entries are frozen; with κ = 0.2, κ_next = 0.5 the code yields
#   P(x_next = x_1) = 0.875 instead of 0.75, so `L_cons` compares velocities at states from different
#   marginals. The test (`test_flow_matching.jl:54-57`) uses `x_1 = 1, x_t = 0` and never hits the branch.
# - **B2** `L_cons` gradient flows through both forward passes (`:292-306`); no stop-gradient on the
#   target branch. Check against EFLOCO eq. 10; as written the two predictions can collapse toward each
#   other. The velocity clamp at ±100 also lets `mse` reach ~10⁴ against an O(1) `l_dfm`.
# - **B3** DFM inference queries `t_idx = 0` on its first step (`:143,193`) while training samples
#   `t ∈ 1:T`.
# - **B4** β endpoints (1e-4, 2e-2) are not scaled with `total_steps` (`get_betas`); with T = 100 (all
#   tests) ᾱ_T ≈ 0.135, so q(x_T | x_0) is far from the Bernoulli(0.5) prior the sampler starts from.
# - **B5** `CosineScheduler`/`LinearScheduler` change only the inference spacing; the noise schedule is
#   always linear (`get_betas` ignores its argument except `total_steps`), contrary to the docstrings
#   ("cosine denoising schedule as used in DIFUSCO", "corruption ratio values from the assigned
#   scheduler").
# - **B6** `Base.convert(::Type{Scheduler{T}}, s)` (`linear.jl:63`, `cosine.jl:64`, `atss.jl:216`) makes
#   `BernoulliDiffusion(CosineScheduler(Float32, …))` silently recompute an Int schedule, contradicting
#   the README's "requires" statements; for ATSS the rounding creates duplicates.
# - **B7** `_embed_timesteps(t_emb, ::Nothing, x_t)` repeats N times even when N per-node timesteps were
#   passed (`diffusion_model.jl:22-26`), although the docstring advertises per-node input.
# - **B8** `feature_dim` is derived independently in `DiffusionModel` (`in_dim − size(x_t,1)`) and
#   `AGNN.feature_dim`; a mismatch silently feeds a time-embedding row as state. No assert.
# - **B9** Samplers hard-code the start time instead of reading `first(scheduler)` /
#   `first(Iterators.reverse(scheduler))`; a scheduler not starting at 0/ending at T silently skips a step.
# - **B10** `Base.collect(s::Scheduler) = s.steps` aliases the internal vector; ATSS never validates
#   `inference_steps < total_steps` and its initial guess may contain duplicates.
# - **B11** `posterior_prob` denominators `0.5(1 − ᾱ_t)` are unguarded for small t (Float32
#   cancellation); output is only clamped afterwards.

# - **B11a** Duplicate inference steps: `CosineScheduler(100, 40)` yields `…, 99, 99, 100` (`cosine.jl:39`),
#   each duplicate a full no-op GNN pass; ATSS Int conversion has no `max(1, …)` floor, so `steps ==
#   [0, 0, …]` is possible and the final Bernoulli step then calls `posterior_prob(…, t=0, s=0)` →
#   `αs[0:0]` → `BoundsError`. `ATSSScheduler.κ̇` is stored but never read. Coordinate descent in
#   `_minimize_atss_schedule` does a single sweep and `sort!(trial)` shifts coordinate indices after an
#   accepted move. A user `κ` returning Float64 silently promotes the whole DFM corruption/velocity path to
#   Float64 on the GPU (`discrete_flow_matching.jl:112`).
