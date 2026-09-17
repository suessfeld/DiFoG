# src/diffusion/methods/diffusion/bernoulli_diffusion.jl
#
# Implements Continuous Diffusion with Bernoulli noise as described in the DIFUSCO paper.
# Handles the forward corruption and backward denoising steps for combinatorial optimization.

using Flux
using Statistics
using Random
using NNlib
using CUDA
using Functors

export BernoulliDiffusion

"""
    BernoulliDiffusion <: Diffusion

Struct to hold parameters for Continuous Diffusion with Bernoulli noise.
Implements the framework described in Section 3.2 of the DIFUSCO paper.
"""
struct BernoulliDiffusion <: Diffusion
    scheduler::Scheduler{Int}
end

Functors.@leaf BernoulliDiffusion

"""
    get_betas(scheduler::Scheduler)

Gets the corruption ratio values from the assigned scheduler (for each step). Length is dependent on the configured stepsize (total_steps) in the scheduler.

Parameters:
    - 'scheduler': The scheduler used.
"""
function get_betas(scheduler::Scheduler)
    T = scheduler.total_steps
    @debug Float32.(range(1f-4, 2f-2, length=T))
    return Float32.(range(1f-4, 2f-2, length=T))
end
 
"""
    get_alphas(scheduler::Scheduler)

Gets the cumulative product of the signal scaling factors for the Bernoulli transition matrices. Length is dependent on the configured stepsize (total_steps) in the scheduler.

Parameters:
    - 'scheduler': The scheduler used.
"""
function get_alphas(scheduler::Scheduler)
    βs = get_betas(scheduler)
    one_minus_2β = 1.0f0 .- 2.0f0 .* βs
    @debug cumprod(one_minus_2β)
    return cumprod(one_minus_2β)
end

"""
    q(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{Int})

Describes one forward step (corruption step) in the diffusion process. See page 4 of DIFUSCO paper.

Parameters:
    - 'diffusion': The diffusion method.
    - 'x_0': The starting solution without any corruption.
    - 't': The timestep which is relevant for the current step. In vector format for parallelization.
"""
function q(diffusion::BernoulliDiffusion, x_0::AbstractArray, t::AbstractVector{<:Integer})
    # FIX: Move the alphas to the GPU BEFORE indexing with the GPU array `t`
    αs = get_alphas(diffusion.scheduler) |> Flux.gpu
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

"""
    corrupt_batch(diffusion::BernoulliDiffusion, x_0::AbstractArray, g::GNNGraph)

Executes the forward corruption process q(x_t | x_0) for a batch of graphs.

Parameters:
    - 'diffusion': The BernoulliDiffusion method.
    - 'x_0': The target solution.
    - 'g': The batch of graphs.
"""
function corrupt_batch(diffusion::BernoulliDiffusion, x_0::AbstractArray, g::GNNGraph)
    T = diffusion.scheduler.total_steps
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


"""
    calculate_loss(::BernoulliDiffusion, preds::AbstractArray, target::AbstractArray)

Calculate the loss function.
    
Parameters:
    - 'preds': The prediction values.
    - 'target': The target values.
"""
function calculate_loss(::BernoulliDiffusion, preds::AbstractArray, target::AbstractArray)
    return Flux.logitbinarycrossentropy(preds, target)
end


"""
    posterior_prob(diffusion::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int, s::Int)

Calculate a vector of probabilities for x_t-1 for a given intermediate solution x_t at timestep t. See eq. (5) and (6) in DIFUSCO.

Parameters:
    - 'diffusion': The diffusion method
    - 'x_t': The intermediate solution which the prediction is based upon.
    - 'x_0_pred': The  prediction for x_0 (by the model).
    - 't': The current timestep.
    - 's': The previous timestep.
"""
function posterior_prob(diffusion::BernoulliDiffusion, x_t::AbstractArray, x_0_pred::AbstractArray, t::Int, s::Int)
    αs = get_alphas(diffusion.scheduler)
    
    ᾱₜ = αs[t:t] |> Flux.gpu
    
    if s > 0
        ᾱₛ = αs[s:s] |> Flux.gpu
    else
        ᾱₛ = Flux.gpu([1.0f0])
    end
    
    α_t_s = ᾱₜ ./ ᾱₛ
    p_t_s = 0.5f0 .* (1.0f0 .+ α_t_s)
    
    # q(x_t | x_{t-1} = 1)
    term_1 = x_t .* p_t_s .+ (1.0f0 .- x_t) .* (1.0f0 .- p_t_s)
    
    # eq 5 evaluated for x_0 = 1: q(x_{t-1}=1 | x_t, x_0=1)
    # => ( q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0=1) ) / q(x_t | x_0=1)
    term_2 = (term_1 .* 0.5f0 .* (1.0f0 .+ ᾱₛ)) ./
             (x_t .* 0.5f0 .* (1.0f0 .+ ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 .- ᾱₜ))
    
    # eq 5 evaluated for x_0 = 0: q(x_{t-1}=1 | x_t, x_0=0)
    # =>( q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0=0) ) / q(x_t | x_0=0)
    term_3 = (term_1 .* 0.5f0 .* (1.0f0 .- ᾱₛ)) ./
             (x_t .* 0.5f0 .* (1.0f0 .- ᾱₜ) .+ (1.0f0 .- x_t) .* 0.5f0 .* (1.0f0 .+ ᾱₜ))
    
    # eq 6 p_θ(x_{t-1}=1 | x_t)
    term_4 = x_0_pred .* term_2 .+ (1.0f0 .- x_0_pred) .* term_3
    
    return clamp.(term_4, 0.0f0, 1.0f0)
end

"""
    p(diffusion::BernoulliDiffusion, model, x_t::AbstractArray, t::Int, s::Int, g::GNNGraph)

Describes one backward (denoising) step for the diffusion process.

Parameters:
    - 'diffusion': The diffusion method
    - 'model': The current model.
    - 'x_t': The current diffusion state of x at the timestep t.
    - 't': The current timestep.
    - 's': The previous (target) timestep for this denoising step.
    - 'g': The provided graphs.
"""
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


"""
    sample_heatmap(diffusion::BernoulliDiffusion, model, shape, g)

Represents an inference call and produces a heatmap for the provided input.

    Parameters:
    - 'diffusion': The diffusion method.
    - 'model': The current model.
    - 'shape': The shape of the graph.
    - 'g': The input graph which will be predicted.
"""
function sample_heatmap(diffusion::BernoulliDiffusion, model, shape::Tuple, g::GNNGraph; kwargs...)
    x_cpu = Float32.(rand(Float32, shape) .> 0.5f0) 
    x = Flux.gpu(x_cpu)
    
    τ_current = diffusion.scheduler.total_steps

    @debug "Scheduler State" (; (f => getfield(diffusion.scheduler, f) for f in fieldnames(typeof(diffusion.scheduler)))...)
    # iterates as defined by the scheduler
    for τ_prev in Iterators.drop(Iterators.reverse(diffusion.scheduler), 1)
        x = p(diffusion, model, x, τ_current, τ_prev, g)
        τ_current = τ_prev
    end

    batch_size = hasproperty(g, :num_graphs) ? g.num_graphs : 1
    t_batch = CUDA.fill(Int32(1), batch_size)

    #GR TODO Diesen Extraaufruf verstehe ich nicht. Wir haben hier als x einen bereits binären Vektor, der nochmals durch das Model geschickt wird um Wahrscheinlichkeiten zu erhalten? Das ergibt für mich keinen Sinn. Sollte nicht term_1 vom letzten Entrauschungsschritt als sinnvolle Heatmap zurückgegeben werden? So wie es aktuell aussieht sind die gemessenen Heatmap-Metriken relativ bedeutungslos. Ich denke, dass das z.B. in DIFUSCO auch nicht so gemacht wird.
    logits = model(x, t_batch, g)
    return Flux.sigmoid.(logits)
end