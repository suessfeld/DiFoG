export CosineScheduler

struct CosineScheduler <: Scheduler
    num_timesteps::Int
    inference_steps::Int
    steps::Vector{Int}
end

function CosineScheduler(num_timesteps::Int, inference_steps::Int)
    c = 1.0 / inference_steps
    steps = [max(1, floor(Int, cos((1.0 - c * i) * (π / 2.0)) * num_timesteps)) for i in 1:inference_steps]
    return CosineScheduler(num_timesteps, inference_steps, steps)
end

function get_betas(s::CosineScheduler)
    s_param = 0.008
    steps_range = 0:s.num_timesteps
    f(t) = cos(((t / s.num_timesteps + s_param) / (1 + s_param)) * (pi / 2))^2
    αs_cum_prod_raw = [f(t) / f(0) for t in steps_range]
    βs = [1 - (αs_cum_prod_raw[t+1] / αs_cum_prod_raw[t]) for t in 1:s.num_timesteps]
    return Float32.(clamp.(βs, 0.0, 0.999))
end