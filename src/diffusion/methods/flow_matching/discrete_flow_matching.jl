using Flux
using Statistics
using Random
using CUDA

export DiscreteFlowMatching

struct DiscreteFlowMatching <: Diffusion
    λ_cons::Float32
    scheduler_power::Float32
    scheduler::Scheduler
end

Flux.@layer DiscreteFlowMatching

function DiscreteFlowMatching(scheduler::Scheduler; λ_cons::Float32=1.0f0,
                                scheduler_power::Float32=1.0f0)
    DiscreteFlowMatching(λ_cons, scheduler_power, scheduler)
end

@inline function kappa(pw::Float32, τ::Real)
    return Float32(clamp(τ, 0f0, 1f0))^pw
end

@inline function kappa_dot(pw::Float32, τ::Real)
    τc = Float32(clamp(τ, 1f-6, 1f0))
    return pw * τc^(pw - 1f0)
end

function velocity_field(diffusion::DiscreteFlowMatching, x_t::AbstractArray,
                         p1t_pred::AbstractArray, τ)
    pw = diffusion.scheduler_power
    κ  = kappa.(pw, τ)
    κ̇  = kappa_dot.(pw, τ)
    denom = max.(1f0 .- κ, 1f-6)
    coef  = κ̇ ./ denom
    return coef .* (p1t_pred .- x_t)
end

_time_shape(x::AbstractArray, n::Int) = ndims(x) == 2 ? (1, n) : (1, 1, n)

function coupled_next_state(diffusion::DiscreteFlowMatching, x_τ::AbstractArray,
                             x_1::AbstractArray, τ, τ_next)
    pw = diffusion.scheduler_power
    κ      = kappa.(pw, τ)
    κ_next = kappa.(pw, τ_next)
    p_commit = clamp.((κ_next .- κ) ./ max.(1f0 .- κ, 1f-6), 0f0, 1f0)

    already_committed = x_τ .== x_1
    
    r = CUDA.rand(Float32, size(x_τ))
    noise = CUDA.rand(Float32, size(x_τ))
    fresh_noise = ifelse.(noise .> 0.5f0, 1.0f0, 0.0f0)

    return ifelse.(already_committed .| (r .< p_commit), x_1, fresh_noise)
end

function sample_timesteps(diffusion::DiscreteFlowMatching, batch_size::Int)
    T = get_num_timesteps(diffusion)
    # Uniform sampling across the training domain
    return rand(Int32(1):Int32(T), batch_size)
end

function corrupt_batch(diffusion::DiscreteFlowMatching, x_1::AbstractArray, g::GNNGraph)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    T = get_num_timesteps(diffusion)

    t_cpu = sample_timesteps(diffusion, batch_size)
    t_device = Flux.gpu(t_cpu) 

    if hasproperty(g, :graph_indicator) && !isnothing(g.graph_indicator)
        t_nodes = t_device[g.graph_indicator]
    else
        num_nodes = size(x_1, 2)
        t_nodes = Flux.gpu(fill(Int32(t_cpu[1]), num_nodes))
    end

    τ_nodes = Float32.(t_nodes) ./ Float32(T)
    shp = _time_shape(x_1, length(τ_nodes))
    
    pw = diffusion.scheduler_power
    κ_reshaped = reshape(kappa.(pw, τ_nodes), shp)

    u = CUDA.rand(Float32, size(x_1))
    mask = u .< κ_reshaped
    
    noise = CUDA.rand(Float32, size(x_1))
    base_noise = ifelse.(noise .> 0.5f0, 1.0f0, 0.0f0)
    
    corrupted_x = ifelse.(mask, x_1, base_noise)

    return corrupted_x, t_device, x_1
end

function calculate_loss(::DiscreteFlowMatching, preds::AbstractArray, target::AbstractArray)
    return Flux.logitbinarycrossentropy(preds, target)
end

function p(diffusion::DiscreteFlowMatching, model, x_τ::AbstractArray, τ::Real, τ_next::Real, g::GNNGraph)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    T = get_num_timesteps(diffusion)
    
    t_idx = clamp(round(Int, τ * T), 0, T)
    t_batch = CUDA.fill(Int32(t_idx), batch_size)

    logits = model(x_τ, t_batch, g)
    p1t_pred = Flux.sigmoid.(logits)

    if τ_next >= 1f0 - 1f-6
        noise = CUDA.rand(Float32, size(x_τ))
        return ifelse.(noise .< p1t_pred, 1.0f0, 0.0f0)
    end

    u_τ = velocity_field(diffusion, x_τ, p1t_pred, Float32(τ))
    x_next_prob = clamp.(x_τ .+ Float32(τ_next - τ) .* u_τ, 0f0, 1f0)
    
    noise = CUDA.rand(Float32, size(x_τ))
    return ifelse.(noise .< x_next_prob, 1.0f0, 0.0f0)
end

function sample_heatmap(diffusion::DiscreteFlowMatching, model, shape::Tuple, g::GNNGraph)
    x_cpu = Float32.(rand(Float32, shape) .> 0.5f0)
    x = Flux.gpu(x_cpu)
    
    T = Float32(get_num_timesteps(diffusion))

    # Flow Matching naturally integrates forwards from 0 to 1
    τ_grid = Float32.(diffusion.scheduler.steps) ./ T
    
    # Enforce boundaries
    τ_grid[1] > 0f0   && pushfirst!(τ_grid, 0f0)
    τ_grid[end] < 1f0 && push!(τ_grid, 1f0)

    for i in 1:length(τ_grid)-1
        x = p(diffusion, model, x, τ_grid[i], τ_grid[i+1], g)
    end

    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = CUDA.fill(Int32(get_num_timesteps(diffusion)), batch_size)
    logits = model(x, t_batch, g)
    return Flux.sigmoid.(logits)
end

function Thesis.compute_loss_and_gradients(gnn::GNN, method::DiscreteFlowMatching,
                                            x_t::AbstractArray, timesteps::AbstractVector,
                                            batch::GNNGraph, x_1::AbstractArray)
    x_dim = size(x_t, 1)
    time_dim = gnn.config.in_dim - x_dim
    gi = batch.graph_indicator
    T = get_num_timesteps(method)

    t_emb = Thesis.get_timestep_embedding(timesteps, time_dim)
    t_emb_nodes = isnothing(gi) ? repeat(t_emb, 1, size(x_t, 2)) : t_emb[:, gi]
    x_input_t = vcat(t_emb_nodes, x_t)

    timesteps_next = min.(T, timesteps .+ 1)
    t_next_emb = Thesis.get_timestep_embedding(timesteps_next, time_dim)
    t_next_emb_nodes = isnothing(gi) ? repeat(t_next_emb, 1, size(x_t, 2)) : t_next_emb[:, gi]

    if isnothing(gi)
        num_nodes = size(x_1, 2)
        τ      = Float32.(timesteps)      ./ Float32(T)
        τ_next = Float32.(timesteps_next) ./ Float32(T)
        τ_nodes      = repeat(τ, num_nodes)
        τ_next_nodes = repeat(τ_next, num_nodes)
    else
        τ      = Float32.(timesteps)      ./ Float32(T)
        τ_next = Float32.(timesteps_next) ./ Float32(T)
        τ_nodes      = τ[gi]
        τ_next_nodes = τ_next[gi]
    end

    shp = _time_shape(x_1, length(τ_nodes))
    τ_r      = reshape(τ_nodes, shp)
    τ_next_r = reshape(τ_next_nodes, shp)

    x_t_next = coupled_next_state(method, x_t, x_1, τ_r, τ_next_r)
    x_input_t_next = vcat(t_next_emb_nodes, x_t_next)

    pw = method.scheduler_power
    denom_t = max.(1f0 .- kappa.(pw, τ_r), 1f-6)
    coef_t = kappa_dot.(pw, τ_r) ./ denom_t
    
    denom_t_next = max.(1f0 .- kappa.(pw, τ_next_r), 1f-6)
    coef_t_next = kappa_dot.(pw, τ_next_r) ./ denom_t_next

    loss, grads = Flux.withgradient(gnn) do gnn_inner
        logits_t = gnn_inner(batch, x_input_t)
        l_dfm = Flux.logitbinarycrossentropy(logits_t, x_1)

        logits_t_next = gnn_inner(batch, x_input_t_next)

        p1t_pred      = Flux.sigmoid.(logits_t)
        p1t_next_pred = Flux.sigmoid.(logits_t_next)

        u_t      = coef_t .* (p1t_pred .- x_t)
        u_t_next = coef_t_next .* (p1t_next_pred .- x_t_next)

        l_cons = Flux.mse(u_t, u_t_next)

        return l_dfm + method.λ_cons * l_cons
    end

    return loss, (Thesis._mat(grads[1]),)
end