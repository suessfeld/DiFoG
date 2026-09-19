# src/utils/plotting.jl
#
# Plotting utilities for visualizing training progress (loss curves) and other diagnostics.

using Plots

export plot_loss_curve

"""
    plot_loss_curve(loss_history::Vector{Float32}, val_loss_history::Vector{Float32}, filepath::AbstractString;
                    title::AbstractString="Training Loss")

Plots per-epoch training loss (and validation loss, if non-empty) and saves it to `filepath`.
The y-axis always starts at 0 and tops out slightly above the highest loss value (10% headroom),
regardless of Plots.jl's automatic scaling, so the curve's shape stays comparable across runs.

Parameters:
    - 'loss_history': Per-epoch average training loss, one entry per epoch (see `train_diffusion_model!`).
    - 'val_loss_history': Per-epoch average validation loss, one entry per epoch, or empty if validation was disabled (`num_val=0`).
    - 'filepath': Output path, including extension.
    - 'title': Plot title.
"""
function plot_loss_curve(loss_history::Vector{Float32}, val_loss_history::Vector{Float32},
                         filepath::AbstractString; title::AbstractString="Training Loss")
    all_losses = isempty(val_loss_history) ? loss_history : vcat(loss_history, val_loss_history)
    y_max = maximum(all_losses) * 1.1f0

    p = Plots.plot(1:length(loss_history), loss_history;
                    label="Train Loss", xlabel="Epoch", ylabel="Loss",
                    title=title, lw=2, legend=:topright, ylims=(0, y_max))

    if !isempty(val_loss_history)
        Plots.plot!(p, 1:length(val_loss_history), val_loss_history; label="Validation Loss", lw=2)
    end

    Plots.savefig(p, filepath)
    return filepath
end