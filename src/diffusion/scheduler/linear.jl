# src/diffusion/scheduler/linear.jl
#
# Linear inference time-step spacing (DIFUSCO §3.3). This only selects WHICH times of the training
# process are visited at inference; it does not change the forward noise schedule β.

export LinearScheduler
using Functors

"""
    LinearScheduler{T} <: TimeSchedule{T}

Represents a linear inference time-step spacing schedule.

Fields:
    - 'total_steps': The total number of steps in the training process.
    - 'inference_steps': The number of inference transitions used.
    - 'steps': The generated sequence of ascending steps (Int for diffusion time, Float32 for progress).
"""
struct LinearScheduler{T} <: TimeSchedule{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
end

Functors.@leaf LinearScheduler

LinearScheduler(total_steps::Int, inference_steps::Int) = LinearScheduler(Int, total_steps, inference_steps)

"""
    _linear_progress(T::Int, M::Int)

Calculates the integer progress points for the linear schedule using integer arithmetic.

Parameters:
    - 'T': The total number of training steps.
    - 'M': The number of inference transitions.
"""
_linear_progress(T::Int, M::Int) =
    M >= T ? collect(0:T) : vcat(0, [max(1, div(T * i, M)) for i in 1:M])

"""
    LinearScheduler(::Type{VT}, total_steps::Int, inference_steps::Int) where {VT<:Union{Int,Float32}}

Constructs a LinearScheduler using the specified type for step representation.

Parameters:
    - 'VT': The target type for the steps (Int or Float32).
    - 'total_steps': The total number of training steps.
    - 'inference_steps': The number of inference transitions.
"""
function LinearScheduler(::Type{VT}, total_steps::Int, inference_steps::Int) where {VT<:Union{Int,Float32}}
    steps = _progress_steps(VT, _linear_progress(total_steps, inference_steps), total_steps)
    @debug "Steps from LinearScheduler: " steps
    return LinearScheduler{VT}(total_steps, inference_steps, steps)
end

Base.convert(::Type{TimeSchedule{T}}, s::LinearScheduler) where {T} = LinearScheduler(T, s.total_steps, s.inference_steps)
Base.convert(::Type{LinearScheduler{T}}, s::LinearScheduler) where {T} = LinearScheduler(T, s.total_steps, s.inference_steps)