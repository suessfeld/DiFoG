# src/utils/tracking.jl
#
# Experiment tracking. The training loop reports its metrics to an `AbstractTracker`: the no-op `NoTracker`
# or the `WandbTracker`, which reports to Weights & Biases (Wandb.jl). Plain-text writers for the histories are
# provided in addition.

export AbstractTracker, NoTracker, WandbTracker, log_metrics!, finish!,
       save_loss_history, save_metric_history

"""
    AbstractTracker

Supertype of all experiment trackers. A tracker implements `log_metrics!(tracker, metrics; step)` and
`finish!(tracker)`.
"""
abstract type AbstractTracker end

"""
    NoTracker()

Tracker that discards all metrics (default of `train_diffusion_model!`).
"""
struct NoTracker <: AbstractTracker end

"""
    log_metrics!(tracker::AbstractTracker, metrics::AbstractDict; step::Integer)

Reports a set of named scalar metrics at a training step (the global batch number).

Parameters:
    - 'tracker': The experiment tracker.
    - 'metrics': Metric names (e.g. "train/loss", "val/gap") mapped to scalar values.
    - 'step': The global batch number the metrics belong to; must not decrease between calls.
"""
log_metrics!(::NoTracker, metrics::AbstractDict; step::Integer) = nothing

"""
    finish!(tracker::AbstractTracker)

Closes the tracker after the last metric of a run was reported.

Parameters:
    - 'tracker': The experiment tracker.
"""
finish!(::AbstractTracker) = nothing

"""
    WandbTracker(; project, name=nothing, config=Dict(), kwargs...)

Tracker that reports to Weights & Biases (one W&B run per tracker). Further keyword arguments (e.g. `entity`,
`group`, `tags`, `mode="offline"`, `dir`) are passed to `wandb.init`.

Parameters:
    - 'project': The W&B project the run is added to.
    - 'name': The display name of the run.
    - 'config': The configuration of the run (hyperparameters), shown and searchable in W&B.
"""
struct WandbTracker <: AbstractTracker
    logger::Wandb.WandbLogger
end

function WandbTracker(; project::AbstractString, name::Union{Nothing,AbstractString}=nothing,
                      config::AbstractDict=Dict{String,Any}(), kwargs...)
    config = Dict{String,Any}(string(k) => v for (k, v) in config)
    # on Windows, Wandb.jl adds a `start_method` setting that current wandb clients reject
    if Sys.iswindows() && !haskey(kwargs, :settings)
        kwargs = (; kwargs..., settings = Wandb.wandb.Settings())
    end
    return WandbTracker(Wandb.WandbLogger(; project, name, config, kwargs...))
end

function log_metrics!(tracker::WandbTracker, metrics::AbstractDict; step::Integer)
    Wandb.log(tracker.logger, Dict{String,Any}(string(k) => Float64(v) for (k, v) in metrics); step=Int(step))
    return nothing
end

function finish!(tracker::WandbTracker)
    close(tracker.logger)
    return nothing
end

"""
    save_loss_history(batch_loss_history::Vector{Float32}, val_loss_history::Vector{Float32},
                      val_loss_batches::Vector{Int}, filepath::AbstractString, batch_size::Int)

Writes the raw loss history of a training run to a plain-text file.

Parameters:
    - 'batch_loss_history': Per-batch training loss across the whole run.
    - 'val_loss_history': Per-epoch average validation loss.
    - 'val_loss_batches': The batch-number at which each validation loss entry was computed.
    - 'filepath': The output path; its extension is replaced by `.txt`.
    - 'batch_size': The training batch size used for this run.
"""
function save_loss_history(batch_loss_history::Vector{Float32}, val_loss_history::Vector{Float32},
                           val_loss_batches::Vector{Int}, filepath::AbstractString, batch_size::Int)
    txt_path = splitext(filepath)[1] * ".txt"

    open(txt_path, "w") do io
        println(io, "# batch_size: ", batch_size)
        println(io, "# batch")
        println(io, "batch,loss")
        for i in eachindex(batch_loss_history)
            println(io, i, ",", batch_loss_history[i])
        end

        println(io, "# validation")
        println(io, "batch,val_loss")
        for i in eachindex(val_loss_history)
            println(io, val_loss_batches[i], ",", val_loss_history[i])
        end
    end

    return txt_path
end

"""
    save_metric_history(filepath::AbstractString, epoch_history::AbstractVector{<:NamedTuple})

Writes the per-epoch metrics of a training run (see `train_diffusion_model!`) as a CSV file with one row per
epoch and one column per metric.

Parameters:
    - 'filepath': The output path of the CSV file.
    - 'epoch_history': One NamedTuple per epoch, all with the same fields.
"""
function save_metric_history(filepath::AbstractString, epoch_history::AbstractVector{<:NamedTuple})
    isempty(epoch_history) && return nothing
    cols = keys(first(epoch_history))
    open(filepath, "w") do io
        println(io, join(cols, ","))
        for row in epoch_history
            println(io, join((row[c] for c in cols), ","))
        end
    end
    return filepath
end
