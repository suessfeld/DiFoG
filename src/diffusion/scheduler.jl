export Scheduler

# can return either ints or floats
abstract type Scheduler{T} end

Base.length(s::Scheduler) = max(0, length(s.steps) - 1)
Base.collect(s::Scheduler) = s.steps # use collect to just get the entire step vector

# Iterator
function Base.iterate(s::Scheduler, state=length(s.steps) - 1)
    if state < 1
        return nothing
    end
    return (s.steps[state], state - 1)
end

# might be useful as well
function get_steps(s::Scheduler{T}) where T
    return s.steps
end