using Flux
using Statistics
using Random
using NNlib

export BernoulliDiffusion

"""
    BernoulliDiffusion

Struct to hold parameters for Discrete Diffusion with Bernoulli noise.
Implements the framework described in Section 3.2 of the DIFUSCO paper.
"""
struct BernoulliDiffusion <: Diffusion
    num_timesteps::Int # defines how many steps the diffusion will take
    βs::AbstractVector{Float32} # defines the corruption ratio for a given timestamp (list of all β's)
    αs::AbstractVector{Float32} # cumulative product of α to skip steps
end

function BernoulliDiffusion(num_timesteps::Int=1000, schedule::String="linear")

        # TODO: maybe make β-bounds adjustable? The parameters are hardcoded from the DIFUSCO paper
        if schedule == "linear"
            scale = 1000 / num_timesteps
            β_1 = scale * 1e-4 
            β_T = scale * 0.02
            βs = range(β_1, β_T, length=num_timesteps) # create list for β values for any given timestamp

        elseif schedule == "cosine"
            s = 0.008
            steps = 0:num_timesteps
            f = t -> cos(((t / num_timesteps + s) / (1 + s)) * (pi / 2))^2
            αs_cum_prod_raw = [f(t) / f(0) for t in steps]
            βs = [1 - (αs_cum_prod_raw[t+1] / αs_cum_prod_raw[t]) for t in 1:num_timesteps]
            βs = clamp.(βs, 0.0, 0.999) # create list for β values for any given timestamp
        else
            error("Unknown schedule: $schedule")
        end
        
        βs = Float32.(βs) 
        one_minus_2β = 1.0f0 .- 2.0f0 .* βs # direct consequence of the matrix Qₜ
        αs = cumprod(one_minus_2β) # precalculate all α values.
        return BernoulliDiffusion(num_timesteps, βs, αs)
    end

Flux.@layer BernoulliDiffusion

"""
    q(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{Int})

Describes one forward step (corruption step) in the diffusion process. See page 4 of DIFUSCO paper.

Parameters:
    - 'diffusion': diffusion method
    - 'x_0': represents the starting solution without any corruption
    - 't': represents the timestep which is relevant for the current step. In vector format for parallelization
"""
function q(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{<:Integer})
    ᾱₜ = diffusion.αs[t]

    nd = ndims(x_0)
    target_shape = nd == 2 ? (1, length(t)) : (1, 1, length(t))
    ᾱₜ_reshaped = reshape(ᾱₜ, target_shape)

    prob_keep = 0.5f0 .* (1.0f0 .+ ᾱₜ_reshaped)

    noise = CUDA.rand(Float32, size(x_0))  # ← explicit CUDA RNG, no deadlock
    flipped = 1.0f0 .- x_0
    should_flip = noise .> prob_keep

    return ifelse.(should_flip, flipped, x_0)
end

"""
    corrupt_batch(diff::BernoulliDiffusion, x_0::AbstractArray, g)

Describes one forward step (corruption step) in the diffusion process. See page 4 of DIFUSCO paper.

Parameters:
    - 'diffusion': diffusion method
    - 'x_0': represents the starting solution without any corruption
    - 'g': a graph
"""
function corrupt_batch(diffusion::BernoulliDiffusion, x_0::AbstractArray, g::GNNGraph)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1

    t_cpu = rand(Int32(1):Int32(diffusion.num_timesteps), batch_size)
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

function compute_loss_value(::BernoulliDiffusion, preds::AbstractArray, target::AbstractArray)
    return Flux.logitbinarycrossentropy(preds, target) # loss
end

"""
    posterior_prob(diff::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int)

Calculate a vector of probabilities for x_t-1 for a given intermediate solution x_t at timestep t. See eq. (5) and (6) in DIFUSCO.

Parameters:
    - 'diffusion': diffusion method
    - 'x_t': the intermediate solution which the prediction is based upon
    - 'x_0_pred': the prediction for x_0 (by the model)
    - 't': the timestep
"""
function posterior_prob(diffusion::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int)
    βₜ = diffusion.βs[t:t] # [t:t] syntax keeps array in GPU
    ᾱₜ = diffusion.αs[t:t]
    
    if t > 1
        ᾱₜ₋₁ = diffusion.αs[t-1:t-1]
    else
        # Match device of αs and fill with 1.0
        ᾱₜ₋₁ = similar(diffusion.αs, 1)
        fill!(ᾱₜ₋₁, 1.0f0)
    end
    
    # q(x_t | x_{t-1} = 1)
    term_1 = x_t .* (1.0f0 .- βₜ) .+ (1.0f0 .- x_t) .* βₜ
    
    # eq 5 evaluated for x_0 = 1: q(x_{t-1}=1 | x_t, x_0=1)
    # => ( q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0=1) ) / q(x_t | x_0=1)
    term_2 = (term_1 .* 0.5f0 .* (1.0f0 .+ ᾱₜ₋₁)) ./ 
                                   (x_t .* 0.5f0 .* (1.0f0 .+ ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 .- ᾱₜ))
                                   
    # eq 5 evaluated for x_0 = 0: q(x_{t-1}=1 | x_t, x_0=0)
    # =>( q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0=0) ) / q(x_t | x_0=0)
    term_3 = (term_1 .* 0.5f0 .* (1.0f0 .- ᾱₜ₋₁)) ./ 
                                   (x_t .* 0.5f0 .* (1.0f0 .- ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 .+ ᾱₜ))
    
    # eq 6 p_θ(x_{t-1}=1 | x_t)
    term_4 = x_0_pred .* term_2 .+ (1.0f0 .- x_0_pred) .* term_3
    
    return clamp.(term_4, 0.0f0, 1.0f0)
end

"""
    p(diffusion::BernoulliDiffusion, model, x_t::AbstractArray, t::Int, g)

Describes one backward (denoising) step for the diffusion process.

Parameters:
    - 'diffusion': diffusion method
    - 'model': the current model
    - 'x_t': the current diffusion state of x at the timestep t
    - 't': the timestep
    - 'g': the graphs
"""
function p(diffusion::BernoulliDiffusion, model, x_t::AbstractArray, t::Int, g)
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = fill(t, batch_size) |> Flux.gpu
    
    logits = model(x_t, t_batch, g)

    # p_θ_x₀₌₁
    term_1 = Flux.sigmoid.(logits)

    # p_θ_xₜ₋₁₌₁
    term2 = posterior_prob(diffusion, x_t, term_1, t)
    
    noise = CUDA.rand(Float32, size(x_t))
    x_t₋₁ = ifelse.(noise .< term2, 1.0f0, 0.0f0)    
    
    return x_t₋₁
end

"""
    sample_heatmap(diffusion::BernoulliDiffusion, model, shape, g)

Represents an inference call and produces a heatmap for the provided input

    Parameters:
    - 'diffusion'diffusion method
    - 'model': the current model
    - 'shape': shape of the graph
    - 'g': The input graph which will be predicted
"""
function sample_heatmap(diffusion::BernoulliDiffusion, model, shape, g)
    # Initialize randomly distributed GPU state 
    x_cpu = Float32.(rand(Float32, shape) .> 0.5f0)
    x = Flux.gpu(x_cpu)

    for t in reverse(2:diffusion.num_timesteps) #TODO: Currrently strictly iterative without any skips (DIFUSCO paper does jumps of 50).
        x = p(diffusion, model, x, t, g)
    end
    
    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = Flux.gpu(fill(1, batch_size))
    
    logits = model(x, t_batch, g)
    return Flux.sigmoid.(logits)
end