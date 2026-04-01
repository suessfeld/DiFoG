using Flux
using Statistics
using Random
using NNlib

export GaussianDiffusion

"""
    GaussianDiffusion

Struct to hold parameters for Continuous Diffusion with Gaussian noise.
Implements the framework described in Section 3.2 of the DIFUSCO paper.
"""
struct GaussianDiffusion <: Diffusion
    num_timesteps::Int
    βs::AbstractVector{Float32} # controls how much noise is added for each step
    αs::AbstractVector{Float32}
    αs_cumprod::AbstractVector{Float32}
    αs_cumprod_prev::AbstractVector{Float32}
    sqrt_αs_cumprod::AbstractVector{Float32}
    sqrt_one_minus_αs_cumprod::AbstractVector{Float32}

    function GaussianDiffusion(num_timesteps::Int=1000, schedule::String="linear")
        if schedule == "linear"
            scale = 1000 / num_timesteps
            beta_start = scale * 1e-4
            beta_end = scale * 0.02
            βs = range(beta_start, beta_end, length=num_timesteps)
        elseif schedule == "cosine"
            s = 0.008
            steps = 0:num_timesteps
            f = t -> cos(((t / num_timesteps + s) / (1 + s)) * (pi / 2))^2
            αs_cumprod = [f(t) / f(0) for t in steps]
            βs = [1 - (αs_cumprod[t+1] / αs_cumprod[t]) for t in 1:num_timesteps]
            βs = clamp.(βs, 0.0, 0.999)
        else
            error("Unknown schedule: $schedule")
        end

        βs = Float32.(βs)
        αs = 1 .- βs
        αs_cumprod = cumprod(αs)
        αs_cumprod_prev = [1.0f0; αs_cumprod[1:end-1]]
        
        sqrt_αs_cumprod = sqrt.(αs_cumprod)
        sqrt_one_minus_αs_cumprod = sqrt.(1 .- αs_cumprod)

        new(num_timesteps, βs, αs, αs_cumprod, αs_cumprod_prev, 
            sqrt_αs_cumprod, sqrt_one_minus_αs_cumprod)
    end
end

Flux.@layer GaussianDiffusion

# corruption step, see page 4 of DIFUSCO
function q_sample(diff::GaussianDiffusion, x_0::AbstractArray, t::AbstractVector{Int}, noise::AbstractArray)
    sqrt_alpha = diff.sqrt_αs_cumprod[t]
    sqrt_one_minus_alpha = diff.sqrt_one_minus_αs_cumprod[t]
    
    nd = ndims(x_0)
    if nd == 2
        target_shape = (1, length(t))
    else
        target_shape = (1, 1, length(t))
    end
    
    sqrt_alpha_r = reshape(sqrt_alpha, target_shape)
    sqrt_one_minus_alpha_r = reshape(sqrt_one_minus_alpha, target_shape)
    
    return sqrt_alpha_r .* x_0 .+ sqrt_one_minus_alpha_r .* noise
end


function compute_loss(diff::GaussianDiffusion, model, x_0::AbstractArray, g)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    
    t = rand(1:diff.num_timesteps, batch_size)
    noise = randn(Float32, size(x_0))
    
    x_lifted = 2.0f0 .* x_0 .- 1.0f0
    
    if hasproperty(g, :graph_indicator) && !isnothing(g.graph_indicator)
        t_nodes = t[g.graph_indicator]
    else
        num_nodes = size(x_0, 2)
        t_nodes = fill(t[1], num_nodes)
    end
    
    x_t = q_sample(diff, x_lifted, t_nodes, noise)

    predicted_noise = model(x_t, t, g)
    
    return Flux.mse(predicted_noise, noise)
end

# denoising step, see page 4 of DIFUSCO
function p_sample(diff::GaussianDiffusion, model, x_t::AbstractArray, t::Int, g)
    β_t = diff.βs[t]
    α_t = diff.αs[t]
    ᾱₜ = diff.αs_cumprod[t]
    
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = fill(t, batch_size)
    
    μ = (1.0f0 / sqrt(α_t)) .* (x_t .- β_t / sqrt(1 - ᾱₜ) .* model(x_t, t_batch, g)) # see page 4 and ho et al. eq (11)
    
    if t > 1
        sigma_t = sqrt(β_t)
        noise = randn(Float32, size(x_t))
        return μ .+ sigma_t .* noise 
    else
        return μ # return only μ at last step.
    end
end

function sample_heatmap(diff::GaussianDiffusion, model, shape, g)
    x = randn(Float32, shape)
    for t in reverse(1:diff.num_timesteps)
        x = p_sample(diff, model, x, t, g)
    end
    return 0.5f0 .* (x .+ 1.0f0)
end