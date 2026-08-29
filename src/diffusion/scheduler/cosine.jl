export CosineScheduler

"""
    CosineScheduler{T} <: Scheduler{T}

Implements the cosine denoising schedule, as used in the DIFUSCO paper.

Fields:
    - `total_steps`: The total number of steps used during the forward training process.
    - `inference_steps`: The desired sequence length for the reverse process, where inference_steps < total_steps.
    - `steps`: The calculated sequence of timesteps τ.
"""
struct CosineScheduler{T} <: Scheduler{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
end

CosineScheduler(total_steps::Int, inference_steps::Int) = CosineScheduler(Int, total_steps, inference_steps)

"""
    CosineScheduler(::Type{Int}, total_steps::Int, inference_steps::Int)

Constructs a CosineScheduler with integer timesteps.
"""
function CosineScheduler(::Type{Int}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        return CosineScheduler{Int}(total_steps, inference_steps, collect(1:total_steps))
    end
    
    c = 1.0 / inference_steps
    steps = [max(1, floor(Int, cos(((1.0 - c * i) * π) / 2.0) * total_steps)) for i in 1:inference_steps]
    @debug "Steps from CosineScheduler: " steps

    return CosineScheduler{Int}(total_steps, inference_steps, steps)
end

"""
    CosineScheduler(::Type{Float32}, total_steps::Int, inference_steps::Int)

Constructs a CosineScheduler with normalized Float timesteps.
"""
function CosineScheduler(::Type{Float32}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        steps = collect(1:total_steps) ./ Float32(total_steps)
        return CosineScheduler{Float32}(total_steps, inference_steps, steps)
    end
    
    c = 1.0 / inference_steps
    steps_int = [max(1, floor(Int, cos(((1.0 - c * i) * π) / 2.0) * total_steps)) for i in 1:inference_steps]
    steps = Float32.(steps_int) ./ Float32(total_steps)
    @debug "Steps from CosineScheduler: " steps
    
    return CosineScheduler{Float32}(total_steps, inference_steps, steps)
end

Base.convert(::Type{Scheduler{T}}, s::CosineScheduler) where {T} = CosineScheduler(T, s.total_steps, s.inference_steps)
Base.convert(::Type{CosineScheduler{T}}, s::CosineScheduler) where {T} = CosineScheduler(T, s.total_steps, s.inference_steps)

function CosineScheduler(total_steps::Int, inference_steps::Int, steps::AbstractVector{T}) where T
    return CosineScheduler{T}(total_steps, inference_steps, Array(steps))
end