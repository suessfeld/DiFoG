using Flux
using Statistics
using Random
using NNlib

export BernoulliDiffusion

struct BernoulliDiffusion <: Diffusion
    num_timesteps::Int # defines how many steps the diffusion will take
    βs::AbstractVector{Float32} # defines how much noise will be introduced at a given timestep (corruption)
    αs::AbstractVector{Float32} # cumulative product of α to skip steps

    function BernoulliDiffusion(num_timesteps::Int=1000, schedule::String="linear")
        # TODO: maybe make β values adjustable? The parameters are chosen from the DIFUSCO paper
        if schedule == "linear"
            scale = 1000 / num_timesteps
            β_1 = scale * 1e-4 
            β_T = scale * 0.02
            β_s = range(β_1, β_T, length=num_timesteps) # create list for β values for any given timestamp
        elseif schedule == "cosine"
            s = 0.008
            steps = 0:num_timesteps
            f = t -> cos(((t / num_timesteps + s) / (1 + s)) * (pi / 2))^2
            αs_cum_prod_raw = [f(t) / f(0) for t in steps]
            β_s = [1 - (αs_cum_prod_raw[t+1] / αs_cum_prod_raw[t]) for t in 1:num_timesteps]
            β_s = clamp.(β_s, 0.0, 0.999) # create list for β values for any given timestamp
        else
            error("Unknown schedule: $schedule")
        end
        
        β_s = Float32.(β_s)
        one_minus_2β = 1.0f0 .- 2.0f0 .* β_s
        cumprod_one_minus_2β = cumprod(one_minus_2β)
        new(num_timesteps, β_s, cumprod_one_minus_2β)
    end
end

Flux.@functor BernoulliDiffusion ()

# corruption step, see page 4 of DIFUSCO
function q_sample(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{Int}) # corruption step, see page 4. of DIFUSCO
    ᾱₜ = diffusion.αs[t]

    nd = ndims(x_0)
    if nd == 2
        target_shape = (1, length(t))
    else
        target_shape = (1, 1, length(t))
    end
    
    ᾱₜ_reshaped = reshape(ᾱₜ, target_shape)
    
    prob_keep = 0.5f0 .* (1.0f0 .+ ᾱₜ_reshaped)
    
    noise = rand(Float32, size(x_0))
    flipped = 1.0f0 .- x_0
    should_flip = noise .> prob_keep
    
    return x_0 .* (.!should_flip) .+ flipped .* should_flip
end

function compute_loss(diff::BernoulliDiffusion, model, x_0::AbstractArray, g)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t = rand(1:diff.num_timesteps, batch_size)

    if hasproperty(g, :graph_indicator) && !isnothing(g.graph_indicator)
        t_nodes = t[g.graph_indicator]
    else
        num_nodes = size(x_0, 2)
        t_nodes = fill(t[1], num_nodes)
    end
    
    x_t = q_sample(diff, x_0, t_nodes)
    
    pred_logits = model(x_t, t, g)
    
    return Flux.logitbinarycrossentropy(pred_logits, x_0)
end

function posterior_prob(diff::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int)
    βₜ = diff.βs[t]
    ᾱₜ = diff.αs[t]
    ᾱₜ₋₁ = t > 1 ? diff.αs[t-1] : 1.0f0
    
    # q(x_t | x_{t-1} = 1)
    term_1 = x_t .* (1.0f0 - βₜ) .+ (1.0f0 .- x_t) .* βₜ
    
    # eq 5 evaluated for x_0 = 1: q(x_{t-1}=1 | x_t, x_0=1)
    # => ( q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0=1) ) / q(x_t | x_0=1)
    term_2 = (term_1 .* 0.5f0 .* (1.0f0 + ᾱₜ₋₁)) ./ 
                                   (x_t .* 0.5f0 .* (1.0f0 + ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 - ᾱₜ))
                                   
    # eq 5 evaluated for x_0 = 0: q(x_{t-1}=1 | x_t, x_0=0)
    # =>( q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0=0) ) / q(x_t | x_0=0)
    term_3 = (term_1 .* 0.5f0 .* (1.0f0 - ᾱₜ₋₁)) ./ 
                                   (x_t .* 0.5f0 .* (1.0f0 - ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 + ᾱₜ))
    
    # eq 6 p_θ(x_{t-1}=1 | x_t)
    term_4 = x_0_pred .* term_2 .+ (1.0f0 .- x_0_pred) .* term_3
    
    return clamp.(term_4, 0.0f0, 1.0f0)
end

# denoising step, see page 4 of DIFUSCO
function p_sample(diff::BernoulliDiffusion, model, x_t::AbstractArray, t::Int, g)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = fill(t, batch_size)
    
    logits = model(x_t, t_batch, g)

    # p_θ_x₀₌₁
    term_1 = Flux.sigmoid.(logits)

    # p_θ_xₜ₋₁₌₁
    term2 = posterior_prob(diff, x_t, term_1, t)
    
    noise = rand(Float32, size(x_t))
    x_t₋₁ = Float32.(noise .< term2)
    
    return x_t₋₁
end

function sample_heatmap(diff::BernoulliDiffusion, model, shape, g)
    x = Float32.(rand(Float32, shape) .> 0.5f0)
    for t in reverse(2:diff.num_timesteps)
        x = p_sample(diff, model, x, t, g)
    end
    
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    logits = model(x, fill(1, batch_size), g)
    return Flux.sigmoid.(logits)
end

# TODO: decide whether this is actually needed. Heatmap sampling is the preferred method.
"""
function sample(diff::BernoulliDiffusion, model, shape, g)
    x = Float32.(rand(Float32, shape) .> 0.5f0)
    for t in reverse(1:diff.num_timesteps)
        x = p_sample(diff, model, x, t, g)
    end
    return x
end
"""