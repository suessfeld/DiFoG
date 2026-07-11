export Scheduler, get_schedule, get_betas, get_num_timesteps, sample_training_timesteps, get_forward_taus

abstract type Scheduler end

function Base.iterate(s::Scheduler, state=length(s.steps) - 1)
    if state < 1
        return nothing
    end
    return (s.steps[state], state - 1)
end

Base.length(s::Scheduler) = max(0, length(s.steps) - 1)
Base.eltype(::Type{<:Scheduler}) = Int

function get_schedule(s::Scheduler)
    return s.steps
end

function get_betas(s::Scheduler)
    error("Method get_betas not implemented for $(typeof(s))")
end

get_num_timesteps(s::Scheduler) = isempty(s.steps) ? 0 : s.steps[end]
get_num_timesteps(d::Diffusion) = get_num_timesteps(d.scheduler)

# --- NEW SCHEDULER LOGIC OUTSOURCED FROM METHODS ---

"""
Default uniform sampling for training.
Can be overridden by specialized schedulers (like ATSS).
"""
function sample_training_timesteps(s::Scheduler, batch_size::Int)
    T = get_num_timesteps(s)
    return rand(Int32(1):Int32(T), batch_size)
end

"""
Generates the forward continuous time sequence τ ∈ [0, 1] for Flow Matching.
Enforces the 0.0 and 1.0 boundaries automatically.
"""
function get_forward_taus(s::Scheduler)
    T = Float32(get_num_timesteps(s))
    τ_grid = Float32.(s.steps) ./ T
    
    # Enforce boundaries natively
    τ_grid[1] > 0f0   && pushfirst!(τ_grid, 0f0)
    τ_grid[end] < 1f0 && push!(τ_grid, 1f0)
    
    return τ_grid
end