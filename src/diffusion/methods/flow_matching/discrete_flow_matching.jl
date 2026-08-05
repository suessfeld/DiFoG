using Flux
using Statistics
using Random
using CUDA

export DiscreteFlowMatching

struct DiscreteFlowMatching <: Diffusion
    scheduler::Scheduler{Float32} # scheduler (ATSS used in original paper)
    λ_cons::Float32 # hyperparameter that balances the consistency loss of the loss function (eq. 11)
    κ::Function # interpolation function of the scheduler
    κ̇ ::Function # derivatibe of κ
end

#eq 7
function uₜ(diffusion::DiscreteFlowMatching, x_t::AbstractArray,
                         p₁ₜ::AbstractArray, t)
    κ_t  = diffusion.κ.(t)
    κ̇_t  = diffusion.κ̇.(t)
    return (κ̇_t ./ max.(1f0 .- κ_t, 1f-6)) .* (p₁ₜ .- x_t)   # eq. 7
end

function coupled_next_state(diffusion::DiscreteFlowMatching, x_t::AbstractArray,
                             x_1::AbstractArray, t, t_next)
    κ      = diffusion.κ.(t)
    κ_next = diffusion.κ.(t_next)
    
    p_commit = clamp.((κ_next .- κ) ./ max.(1f0 .- κ, 1f-6), 0f0, 1f0)

    already_committed = x_t .== x_1
    
    r = CUDA.rand(Float32, size(x_t))
    noise = CUDA.rand(Float32, size(x_t))
    fresh_noise = ifelse.(noise .> 0.5f0, 1.0f0, 0.0f0)

    return ifelse.(already_committed .| (r .< p_commit), x_1, fresh_noise)
end

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

    # was: κ_reshaped = reshape(kappa.(pw, τ_nodes), shp)  — `kappa`/`scheduler_power` don't exist;
    # use the diffusion's own κ::Function field directly, consistent with uₜ/coupled_next_state
    κ_reshaped = reshape(diffusion.κ.(τ_nodes), shp)

    u = CUDA.rand(Float32, size(x_1))
    mask = u .< κ_reshaped

    noise = CUDA.rand(Float32, size(x_1))
    base_noise = ifelse.(noise .> 0.5f0, 1.0f0, 0.0f0)

    corrupted_x = ifelse.(mask, x_1, base_noise)

    return corrupted_x, t_device, x_1
end

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
    x_next_prob = euler_step(x_τ, u_τ, τ, τ_next)                # Eq. 6

    noise = CUDA.rand(Float32, size(x_τ))
    return ifelse.(noise .< x_next_prob, 1.0f0, 0.0f0)
end

function euler_step(x_t::AbstractArray, u_t::AbstractArray, τ::Real, τ_next::Real)
    h = Float32(τ_next - τ)
    return clamp.(x_t .+ h .* u_t, 0f0, 1f0)
end

function sample_heatmap(diffusion::DiscreteFlowMatching, model, shape::Tuple, g::GNNGraph; kwargs...)
    x_cpu = Float32.(rand(Float32, shape) .> 0.5f0)
    x = Flux.gpu(x_cpu)
    
    τ_current = 0.0f0

    @info "Scheduler State" (; (f => getfield(diffusion.scheduler, f) for f in fieldnames(typeof(diffusion.scheduler)))...)
    for τ_step in diffusion.scheduler
        x = p(diffusion, model, x, τ_current, τ_step, g)
        τ_current = τ_step   # was: τ_current = τ_prev  (undefined variable)
    end

    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = CUDA.fill(Int32(diffusion.scheduler.total_steps), batch_size)

    logits = model(x, t_batch, g)
    return Flux.sigmoid.(logits)
end

function L_dfm(logits::AbstractArray, x_1::AbstractArray)
    return Flux.logitbinarycrossentropy(logits, x_1)
end

function L_total(l_dfm, l_cons, λ_cons::Float32)
    return l_dfm + λ_cons * l_cons
end

function L_cons(u_t::AbstractArray, u_t_next::AbstractArray)
    return Flux.mse(u_t, u_t_next)
end

function _build_model_input(gnn, x::AbstractArray, timesteps::AbstractVector, gi, time_dim::Int)
    t_emb = Thesis.get_timestep_embedding(timesteps, time_dim)
    t_emb_nodes = isnothing(gi) ? repeat(t_emb, 1, size(x, 2)) : t_emb[:, gi]
    return vcat(t_emb_nodes, x)
end

function _nodewise_tau(timesteps::AbstractVector, T::Int, gi, num_nodes::Int, shp)
    τ = Float32.(timesteps) ./ Float32(T)
    τ_nodes = isnothing(gi) ? repeat(τ, num_nodes) : τ[gi]
    return reshape(τ_nodes, shp)
end

_time_shape(x::AbstractArray, n::Int) = ndims(x) == 2 ? (1, n) : (1, 1, n)

function Thesis.compute_loss_and_gradients(gnn::GNN, method::DiscreteFlowMatching,
                                            x_t::AbstractArray, timesteps::AbstractVector,
                                            batch::GNNGraph, x_1::AbstractArray)
    x_dim = size(x_t, 1)
    time_dim = gnn.config.in_dim - x_dim
    gi = batch.graph_indicator
    T = method.scheduler.total_steps
    
    x_input_t = _build_model_input(gnn, x_t, timesteps, gi, time_dim)

    timesteps_next = min.(T, timesteps .+ 1)
    shp = _time_shape(x_1, isnothing(gi) ? size(x_1, 2) * length(timesteps) : length(gi))

    τ_r      = _nodewise_tau(timesteps, T, gi, size(x_1, 2), shp)
    τ_next_r = _nodewise_tau(timesteps_next, T, gi, size(x_1, 2), shp)

    x_t_next = coupled_next_state(method, x_t, x_1, τ_r, τ_next_r)
    x_input_t_next = _build_model_input(gnn, x_t_next, timesteps_next, gi, time_dim)

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