using Flux
using Statistics
using Random
using NNlib
using CUDA

export BernoulliDiffusion

struct BernoulliDiffusion <: Diffusion
    scheduler::Scheduler
end

Flux.@layer BernoulliDiffusion

function get_alphas(diffusion::BernoulliDiffusion)
    βs = get_betas(diffusion.scheduler)
    one_minus_2β = 1.0f0 .- 2.0f0 .* βs
    return cumprod(one_minus_2β)
end

function q(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{<:Integer})
    # FIX: Move the alphas to the GPU BEFORE indexing with the GPU array `t`
    αs = get_alphas(diffusion) |> Flux.gpu
    ᾱₜ = αs[t]
    
    nd = ndims(x_0)
    target_shape = nd == 2 ? (1, length(t)) : (1, 1, length(t))
    ᾱₜ_reshaped = reshape(ᾱₜ, target_shape)
    
    prob_keep = 0.5f0 .* (1.0f0 .+ ᾱₜ_reshaped)
    noise = CUDA.rand(Float32, size(x_0))
    flipped = 1.0f0 .- x_0
    should_flip = noise .> prob_keep
    
    return ifelse.(should_flip, flipped, x_0)
end

function corrupt_batch(diffusion::BernoulliDiffusion, x_0::AbstractArray, g::GNNGraph)
    T = get_num_timesteps(diffusion)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    
    t_cpu = rand(Int32(1):Int32(T), batch_size)
    t_device = Flux.gpu(t_cpu)
    
    if hasproperty(g, :graph_indicator) && !isnothing(g.graph_indicator)
        t_nodes = t_device[g.graph_indicator]
    else
        num_nodes = size(x_0, 2)
        t_nodes = Flux.gpu(fill(Int32(t_cpu[1]), num_nodes))
    end
    
    corrupted_x = q(diffusion, x_0, t_nodes)
    return corrupted_x, t_device, x_0
end

function calculate_loss(::BernoulliDiffusion, preds::AbstractArray, target::AbstractArray)
    return Flux.logitbinarycrossentropy(preds, target)
end

function posterior_prob(diffusion::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int, s::Int)
    αs = get_alphas(diffusion)
    
    # FIX: Move the sliced alphas to the GPU so they can do math with x_t
    ᾱₜ = αs[t:t] |> Flux.gpu
    
    if s > 0
        ᾱₛ = αs[s:s] |> Flux.gpu
    else
        ᾱₛ = Flux.gpu([1.0f0])
    end
    
    α_t_s = ᾱₜ ./ ᾱₛ
    p_t_s = 0.5f0 .* (1.0f0 .+ α_t_s)
    
    term_1 = x_t .* p_t_s .+ (1.0f0 .- x_t) .* (1.0f0 .- p_t_s)
    
    term_2 = (term_1 .* 0.5f0 .* (1.0f0 .+ ᾱₛ)) ./
             (x_t .* 0.5f0 .* (1.0f0 .+ ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 .- ᾱₜ))
             
    term_3 = (term_1 .* 0.5f0 .* (1.0f0 .- ᾱₛ)) ./
             (x_t .* 0.5f0 .* (1.0f0 .- ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 .+ ᾱₜ))
             
    term_4 = x_0_pred .* term_2 .+ (1.0f0 .- x_0_pred) .* term_3
    
    return clamp.(term_4, 0.0f0, 1.0f0)
end

function p(diffusion::BernoulliDiffusion, model, x_t::AbstractArray, t::Int, s::Int, g::GNNGraph)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = CUDA.fill(Int32(t), batch_size)
    
    logits = model(x_t, t_batch, g)
    term_1 = Flux.sigmoid.(logits)
    term2 = posterior_prob(diffusion, x_t, term_1, t, s)
    
    noise = CUDA.rand(Float32, size(x_t))
    x_s = ifelse.(noise .< term2, 1.0f0, 0.0f0)
    
    return x_s
end

function sample_heatmap(diffusion::BernoulliDiffusion, model, shape::Tuple, g::GNNGraph; kwargs...)
    x_cpu = Float32.(rand(Float32, shape) .> 0.5f0) 
    x = Flux.gpu(x_cpu)
    
    # Initialize state at T (the highest step)
    τ_current = get_num_timesteps(diffusion)

    # Natively iterates T-1, T-2, ..., 1
    for τ_prev in diffusion.scheduler
        x = p(diffusion, model, x, τ_current, τ_prev, g)
        τ_current = τ_prev
    end

    # The final step down to 0
    x = p(diffusion, model, x, τ_current, 0, g)

    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = CUDA.fill(Int32(1), batch_size)

    logits = model(x, t_batch, g)
    return Flux.sigmoid.(logits)
end