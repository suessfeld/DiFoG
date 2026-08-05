export CosineScheduler

struct CosineScheduler{T} <: Scheduler{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
end

CosineScheduler(total_steps::Int, inference_steps::Int) = CosineScheduler(Int, total_steps, inference_steps)

function CosineScheduler(::Type{Int}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        return CosineScheduler{Int}(total_steps, inference_steps, collect(1:total_steps))
    end
    
    c = 1.0 / inference_steps
    steps = [max(1, floor(Int, cos(((1.0 - c * i) * π) / 2.0) * total_steps)) for i in 1:inference_steps]
    return CosineScheduler{Int}(total_steps, inference_steps, steps)
end

function CosineScheduler(::Type{Float32}, total_steps::Int, inference_steps::Int)
    if inference_steps >= total_steps
        steps = collect(1:total_steps) ./ Float32(total_steps)
        return CosineScheduler{Float32}(total_steps, inference_steps, steps)
    end
    
    c = 1.0 / inference_steps
    steps_int = [max(1, floor(Int, cos(((1.0 - c * i) * π) / 2.0) * total_steps)) for i in 1:inference_steps]
    steps = Float32.(steps_int) ./ Float32(total_steps)
    return CosineScheduler{Float32}(total_steps, inference_steps, steps)
end

Base.convert(::Type{Scheduler{T}}, s::CosineScheduler) where {T} = CosineScheduler(T, s.total_steps, s.inference_steps)
Base.convert(::Type{CosineScheduler{T}}, s::CosineScheduler) where {T} = CosineScheduler(T, s.total_steps, s.inference_steps)

function CosineScheduler(total_steps::Int, inference_steps::Int, steps::AbstractVector{T}) where T
    return CosineScheduler{T}(total_steps, inference_steps, Array(steps))
end