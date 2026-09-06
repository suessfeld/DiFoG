# src/diffusion/scheduler.jl
#
# Defines the abstract Scheduler type and basic iteration utilities for managing denoising and integration timesteps across different methods.

export Scheduler

"""
    Scheduler{T}

An abstract implementation of a scheduler. Implementations can either be of type Int or Float32 depending on the chosen diffusion method. It implements the iterator pattern, thus a Scheduler is designed to be iterated upon, returning the defined scheduling steps.    
"""
abstract type Scheduler{T} end

Base.length(s::Scheduler) = max(0, length(s.steps))
Base.collect(s::Scheduler) = s.steps # use collect to just get the entire step vector
Base.eltype(::Type{<:Scheduler{T}}) where T = T # Just added for correct typing - does not really have an impact

# Forward Iterator τ_0 → τ_T
function Base.iterate(s::Scheduler, state::Int=1)
    state > length(s.steps) && return nothing
    return (s.steps[state], state + 1)
end

# Reverse view: τ_T → τ_0
function Base.iterate(r::Iterators.Reverse{<:Scheduler}, state::Int=length(r.itr.steps))
    state < 1 && return nothing
    return (r.itr.steps[state], state - 1)
end
Base.length(r::Iterators.Reverse{<:Scheduler}) = length(r.itr.steps)

function get_steps(s::Scheduler{T}) where T
    return s.steps
end