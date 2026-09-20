# src/utils/plotting.jl
#
# Plotting utilities for visualizing training progress (loss curves) and other diagnostics.

using Plots
using Statistics

export plot_loss_curve

"""
    plot_loss_curve(batch_loss_history::Vector{Float32}, val_loss_history::Vector{Float32},
                    val_loss_batches::Vector{Int}, filepath::AbstractString, batch_size::Int;
                    title::AbstractString="Training Loss", smoothing_window::Int=25)

Plots per-batch training loss (raw, plus a moving-average smoothed line) and per-epoch validation
loss, and saves it to `filepath`. Batch-level tracking is used instead of epoch-averaging so fast
early convergence within an epoch is visible rather than hidden by the epoch average (see project
notes — an epoch-averaged plot previously looked flat from epoch 1 even though the model had
actually converged within the first few dozen batches).

The y-axis always starts at 0 and tops out slightly above the highest loss value (10% headroom).

Parameters:
    - 'batch_loss_history': Per-batch training loss, one entry per batch, across the whole run.
    - 'val_loss_history': Per-epoch average validation loss, or empty if validation was disabled.
    - 'val_loss_batches': The batch-number at which each `val_loss_history` entry was computed
      (validation still runs once per epoch — this places those points at the correct x-position
      on the shared batch-number axis, not one point per batch).
    - 'filepath': Output path, including extension.
    - 'batch_size': The training batch size used for this run — passed through to
      `save_loss_history` so epoch boundaries can be reconstructed from batch number later.
    - 'title': Plot title.
    - 'smoothing_window': Window size (in batches) for the moving-average overlay on the raw
      training curve. Clamped to the actual history length if shorter.
"""
function plot_loss_curve(batch_loss_history::Vector{Float32}, val_loss_history::Vector{Float32},
                         val_loss_batches::Vector{Int}, filepath::AbstractString, batch_size::Int;
                         title::AbstractString="Training Loss", smoothing_window::Int=25)
    batches = 1:length(batch_loss_history)
    all_losses = isempty(val_loss_history) ? batch_loss_history : vcat(batch_loss_history, val_loss_history)
    y_max = maximum(all_losses) * 1.1f0

    p = Plots.plot(batches, batch_loss_history;
                    label="Train Loss (raw)", xlabel="Batch", ylabel="Loss",
                    title=title, lw=1, alpha=0.3, color=:steelblue, legend=:topright, ylims=(0, y_max))

    if length(batch_loss_history) >= 2
        w = min(smoothing_window, length(batch_loss_history))
        smoothed = [mean(batch_loss_history[max(1, i - w + 1):i]) for i in 1:length(batch_loss_history)]
        Plots.plot!(p, batches, smoothed; label="Train Loss (smoothed, w=$w)", lw=2, color=:steelblue)
    end

    if !isempty(val_loss_history)
        Plots.plot!(p, val_loss_batches, val_loss_history;
                    label="Validation Loss", lw=2, marker=:circle, color=:darkorange)
    end

    Plots.savefig(p, filepath)
    save_loss_history(batch_loss_history, val_loss_history, val_loss_batches, filepath, batch_size)
    return filepath
end

"""
    save_loss_history(batch_loss_history::Vector{Float32}, val_loss_history::Vector{Float32},
                      val_loss_batches::Vector{Int}, filepath::AbstractString, batch_size::Int)

Writes the raw loss history backing `plot_loss_curve` to a plain-text CSV, saved next to the PNG
(same path, `.txt` extension) so plots can be regenerated or reworked later — e.g. for the thesis
evaluation chapter — without rerunning training.

Two sections, since training loss is per-batch but validation loss is only per-epoch (see
`train_diffusion_model!`): a `batch` table (one row per training batch) and a `validation` table
(one row per epoch, indexed by the batch number it was measured at via `val_loss_batches`).
`batch_size` is written as a header comment, not a column — needed to reconstruct epoch boundaries
from batch number later, since the batch history itself only tracks a running count, not epoch.

Parameters:
    - 'batch_loss_history', 'val_loss_history', 'val_loss_batches': same as `plot_loss_curve`.
    - 'filepath': the PNG path this history belongs to — the `.txt` is written alongside it (same
      stem, `.txt` extension), not at this exact path.
    - 'batch_size': the training batch size used for this run.
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