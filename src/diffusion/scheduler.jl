# src/diffusion/scheduler.jl
#
# Defines the abstract TimeSchedule type, the shared time convention of all schedules, and basic
# iteration utilities for managing denoising and integration timesteps across different methods.

export TimeSchedule

"""
    TimeSchedule{T}

An abstract base type for schedulers. Implementations use either Int or Float32 step types and support the iterator pattern to yield scheduling steps.

Parameters:
    - 'T': The data type of the scheduling steps (e.g., Int or Float32).
"""
abstract type TimeSchedule{T} end

Base.length(s::TimeSchedule) = length(s.steps)
Base.collect(s::TimeSchedule) = copy(s.steps)
Base.eltype(::Type{<:TimeSchedule{T}}) where T = T
function Base.iterate(s::TimeSchedule, state::Int=1)
    state > length(s.steps) && return nothing
    return (s.steps[state], state + 1)
end

# Reverse view
function Base.iterate(r::Iterators.Reverse{<:TimeSchedule}, state::Int=length(r.itr.steps))
    state < 1 && return nothing
    return (r.itr.steps[state], state - 1)
end
Base.length(r::Iterators.Reverse{<:TimeSchedule}) = length(r.itr.steps)

"""
    _progress_steps(::Type{Int}, progress::AbstractVector{<:Integer}, T::Int)
    _progress_steps(::Type{Float32}, progress::AbstractVector{<:Integer}, T::Int)

Maps integer progress points to their stored representation (either Float32 progress or Int diffusion times). The resulting steps are sorted in ascending order and deduplicated.

Parameters:
    - 'Type': The target type for the steps (Int or Float32).
    - 'progress': A vector of integer progress points.
    - 'T': The total number of steps in the training process.
"""
_progress_steps(::Type{Int}, progress::AbstractVector{<:Integer}, T::Int) = sort!(unique!(T .- collect(progress)))
_progress_steps(::Type{Float32}, progress::AbstractVector{<:Integer}, T::Int) =
    Float32.(sort!(unique!(collect(progress)))) ./ Float32(T)