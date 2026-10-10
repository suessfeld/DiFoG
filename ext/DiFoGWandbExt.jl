# ext/DiFoGWandbExt.jl
#
# Package extension, loaded when Wandb.jl is loaded next to DiFoG: experiment tracking with Weights & Biases.

module DiFoGWandbExt

using DiFoG
using Wandb

"""
    WandbRun

Tracker reporting to one Weights & Biases run. Created with `DiFoG.WandbTracker`.
"""
struct WandbRun <: DiFoG.AbstractTracker
    logger::WandbLogger
end

function DiFoG.WandbTracker(; project::AbstractString, name::Union{Nothing,AbstractString}=nothing,
                            config::AbstractDict=Dict{String,Any}(), kwargs...)
    config = Dict{String,Any}(string(k) => v for (k, v) in config)
    # on Windows, Wandb.jl adds a `start_method` setting that current wandb clients reject
    if Sys.iswindows() && !haskey(kwargs, :settings)
        kwargs = (; kwargs..., settings = Wandb.wandb.Settings())
    end
    return WandbRun(WandbLogger(; project, name, config, kwargs...))
end

function DiFoG.log_metrics!(tracker::WandbRun, metrics::AbstractDict; step::Integer)
    Wandb.log(tracker.logger, Dict{String,Any}(string(k) => Float64(v) for (k, v) in metrics); step=Int(step))
    return nothing
end

function DiFoG.finish!(tracker::WandbRun)
    close(tracker.logger)
    return nothing
end

end
