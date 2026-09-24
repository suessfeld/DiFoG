# src/diffusion/scheduler/cosine.jl
#
# Cosine inference time-step spacing (DIFUSCO §3.3). This only selects WHICH times of the training
# process are visited at inference; it does not change the forward noise schedule β (always the linear
# DIFUSCO β, see `get_betas`).

export CosineScheduler
using Functors

"""
    CosineScheduler{T} <: TimeSchedule{T}

Represents a cosine inference time-step spacing schedule, concentrating steps in the low-noise regime.

Fields:
    - 'total_steps': The total number of steps in the training process.
    - 'inference_steps': The number of transitions used during inference.
    - 'steps': The generated sequence of ascending steps (Int for diffusion time, Float32 for progress).
"""
struct CosineScheduler{T} <: TimeSchedule{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
end

Functors.@leaf CosineScheduler

CosineScheduler(total_steps::Int, inference_steps::Int) = CosineScheduler(Int, total_steps, inference_steps)

"""
    _cosine_progress(T::Int, M::Int)

Calculates the integer progress points for the cosine schedule.

Parameters:
    - 'T': The total number of training steps.
    - 'M': The number of inference transitions.
"""
function _cosine_progress(T::Int, M::Int)
    M >= T && return collect(0:T)
    c = 1.0 / M
    # p_M = T set explicitly; the cos expression itself is unchanged so no value shifts through floor
    return vcat(0, [max(1, floor(Int, cos(((1.0 - c * i) * π) / 2.0) * T)) for i in 1:M-1], T)
end

"""
    CosineScheduler(::Type{VT}, total_steps::Int, inference_steps::Int) where {VT<:Union{Int,Float32}}

Constructs a CosineScheduler using the specified type for step representation.

Parameters:
    - 'VT': The target type for the steps (Int or Float32).
    - 'total_steps': The total number of training steps.
    - 'inference_steps': The number of inference transitions.
"""
function CosineScheduler(::Type{VT}, total_steps::Int, inference_steps::Int) where {VT<:Union{Int,Float32}}
    steps = _progress_steps(VT, _cosine_progress(total_steps, inference_steps), total_steps)
    @debug "Steps from CosineScheduler: " steps
    return CosineScheduler{VT}(total_steps, inference_steps, steps)
end

Base.convert(::Type{TimeSchedule{T}}, s::CosineScheduler) where {T} = CosineScheduler(T, s.total_steps, s.inference_steps)
Base.convert(::Type{CosineScheduler{T}}, s::CosineScheduler) where {T} = CosineScheduler(T, s.total_steps, s.inference_steps)