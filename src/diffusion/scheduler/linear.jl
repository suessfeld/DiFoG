# src/diffusion/scheduler/linear.jl
#
# Implements a linear denoising schedule to control the noise ratios during the forward and reverse diffusion processes.

export LinearScheduler

"""
    LinearScheduler{T} <: Scheduler{T}

Implements the linear denoising schedule, as used in the DIFUSCO paper.

Fields:
    - `total_steps`: The total number of steps used during the forward training process.
    - `inference_steps`: The desired sequence length for the reverse process, where inference_steps < total_steps.
    - `steps`: The calculated sequence of timesteps τ.
"""
struct LinearScheduler{T} <: Scheduler{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
end

LinearScheduler(total_steps::Int, inference_steps::Int) = LinearScheduler(Int, total_steps, inference_steps)

"""
    LinearScheduler(::Type{Int}, total_steps::Int, inference_steps::Int)

Constructs a LinearScheduler with integer timesteps.
"""
function LinearScheduler(::Type{Int}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        return LinearScheduler{Int}(total_steps, inference_steps, collect(0:total_steps))
    end

    c = Float64(total_steps) / Float64(inference_steps)
    steps = vcat(0, [max(1, floor(Int, c * i)) for i in 1:inference_steps])  # τ_0 = 0 prepended
    @debug "Steps from LinearScheduler: " steps

    return LinearScheduler{Int}(total_steps, inference_steps, steps)
end

"""
    LinearScheduler(::Type{Float32}, total_steps::Int, inference_steps::Int)

Constructs a LinearScheduler with normalized Float timesteps.
"""
function LinearScheduler(::Type{Float32}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        steps = collect(0:total_steps) ./ Float32(total_steps)
        return LinearScheduler{Float32}(total_steps, inference_steps, steps)
    end

    c = Float64(total_steps) / Float64(inference_steps)
    steps_int = [max(1, floor(Int, c * i)) for i in 1:inference_steps]
    steps = vcat(0.0f0, Float32.(steps_int) ./ Float32(total_steps))  # τ_0 = 0 prepended
    @debug "Steps from LinearScheduler: " steps

    return LinearScheduler{Float32}(total_steps, inference_steps, steps)
end

Base.convert(::Type{Scheduler{T}}, s::LinearScheduler) where {T} = LinearScheduler(T, s.total_steps, s.inference_steps)
Base.convert(::Type{LinearScheduler{T}}, s::LinearScheduler) where {T} = LinearScheduler(T, s.total_steps, s.inference_steps)

# Catch GPU arrays during Flux.gpu traversal and force them back to the CPU
function LinearScheduler(total_steps::Int, inference_steps::Int, steps::AbstractVector{T}) where T
    return LinearScheduler{T}(total_steps, inference_steps, Array(steps))
end