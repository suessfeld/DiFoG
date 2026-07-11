export ATSSScheduler

struct ATSSScheduler <: Scheduler
    num_timesteps::Int
    inference_steps::Int
    steps::Vector{Int}
end

function get_betas(s::ATSSScheduler)
    s_param = 0.008
    steps_range = 0:s.num_timesteps
    f(t) = cos(((t / s.num_timesteps + s_param) / (1 + s_param)) * (pi / 2))^2
    αs_cum_prod_raw = [f(t) / f(0) for t in steps_range]
    βs = [1 - (αs_cum_prod_raw[t+1] / αs_cum_prod_raw[t]) for t in 1:s.num_timesteps]
    return Float32.(clamp.(βs, 0.0, 0.999))
end

# Override the default training sampling with ATSS logic
function sample_training_timesteps(s::ATSSScheduler, batch_size::Int)
    sched = s.steps
    if length(sched) < 2
        return rand(Int32(1):Int32(s.num_timesteps), batch_size)
    end
    
    n_intervals = length(sched) - 1
    t_out = Vector{Int32}(undef, batch_size)
    for b in 1:batch_size
        n = rand(1:n_intervals)
        lo, hi = max(sched[n], 1), max(sched[n+1], sched[n])
        t_out[b] = Int32(rand(lo:hi))
    end
    return t_out
end