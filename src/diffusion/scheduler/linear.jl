export LinearScheduler

struct LinearScheduler{T} <: Scheduler{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
end

LinearScheduler(total_steps::Int, inference_steps::Int) = LinearScheduler(Int, total_steps, inference_steps)

function LinearScheduler(::Type{Int}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        return LinearScheduler{Int}(total_steps, inference_steps, collect(1:total_steps))
    end
    
    c = Float64(total_steps) / Float64(inference_steps)
    steps = [max(1, floor(Int, c * i)) for i in 1:inference_steps]
    return LinearScheduler{Int}(total_steps, inference_steps, steps)
end

function LinearScheduler(::Type{Float32}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        steps = collect(1:total_steps) ./ Float32(total_steps)
        return LinearScheduler{Float32}(total_steps, inference_steps, steps)
    end
    
    c = Float64(total_steps) / Float64(inference_steps)
    steps_int = [max(1, floor(Int, c * i)) for i in 1:inference_steps]
    steps = Float32.(steps_int) ./ Float32(total_steps)
    return LinearScheduler{Float32}(total_steps, inference_steps, steps)
end

Base.convert(::Type{Scheduler{T}}, s::LinearScheduler) where {T} = LinearScheduler(T, s.total_steps, s.inference_steps)
Base.convert(::Type{LinearScheduler{T}}, s::LinearScheduler) where {T} = LinearScheduler(T, s.total_steps, s.inference_steps)

# Catch GPU arrays during Flux.gpu traversal and force them back to the CPU
function LinearScheduler(total_steps::Int, inference_steps::Int, steps::AbstractVector{T}) where T
    return LinearScheduler{T}(total_steps, inference_steps, Array(steps))
end