export LinearScheduler

struct LinearScheduler <: Scheduler
    num_timesteps::Int
    inference_steps::Int
    steps::Vector{Int}
end

function LinearScheduler(num_timesteps::Int, inference_steps::Int)
    c = num_timesteps / inference_steps
    steps = [max(1, floor(Int, c * i)) for i in 1:inference_steps]
    return LinearScheduler(num_timesteps, inference_steps, steps)
end

function get_betas(s::LinearScheduler)
    scale = 1000 / s.num_timesteps
    β_1 = scale * 1e-4
    β_T = scale * 0.02
    return Float32.(range(β_1, β_T, length=s.num_timesteps))
end