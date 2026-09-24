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

Plots per-batch training loss with a moving-average overlay and per-epoch validation loss, saving the result to the specified filepath.

Parameters:
    - 'batch_loss_history': Per-batch training loss, one entry per batch, across the whole run.
    - 'val_loss_history': Per-epoch average validation loss, or empty if validation was disabled.
    - 'val_loss_batches': The batch-number at which each validation loss entry was computed.
    - 'filepath': Output path for the plot, including extension.
    - 'batch_size': The training batch size used for this run.
    - 'title': The title of the plot.
    - 'smoothing_window': The window size (in batches) for the moving-average overlay.
"""
function plot_loss_curve(batch_loss_history::Vector{Float32}, val_loss_history::Vector{Float32},
                         val_loss_batches::Vector{Int}, filepath::AbstractString, batch_size::Int;
                         title::AbstractString="Training Loss", smoothing_window::Int=25)
    batches = 1:length(batch_loss_history)
    all_losses = vcat(batch_loss_history, val_loss_history)
    finite_losses = filter(isfinite, all_losses)
    y_max = isempty(finite_losses) ? 1f0 : max(maximum(finite_losses) * 1.1f0, 1f-6)

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

Writes the raw loss history backing the loss curve plot to a plain-text file alongside the generated image.

Parameters:
    - 'batch_loss_history': Per-batch training loss across the whole run.
    - 'val_loss_history': Per-epoch average validation loss.
    - 'val_loss_batches': The batch-number at which each validation loss entry was computed.
    - 'filepath': The output path of the corresponding plot (used to determine the text file path).
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