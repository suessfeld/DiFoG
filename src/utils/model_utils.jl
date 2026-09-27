# src/utils/model_utils.jl
#
# Utilities for safely saving and loading trained GNN model states to and from disk using JLD2.
# Next to the weights, a training fingerprint (`meta`: label solver, method, dataset, hyperparameters) is stored,
# so a model file trained under a different configuration is not silently reused under the same file name.

using Flux
using JLD2

export save_model, load_model!

"""
    save_model(model::DiffusionModel, filepath::String; meta::Dict{String,String}=Dict{String,String}())

Saves the state of the GNN model to a JLD2 file, together with the training fingerprint `meta`. Moves the model
to CPU before saving.

Parameters:
    - 'model': The model to be saved.
    - 'filepath': The filepath where the model should be saved.
    - 'meta': The training fingerprint (configuration the weights were trained with).
"""
function save_model(model::DiffusionModel, filepath::String; meta::Dict{String,String}=Dict{String,String}())
    model_state = Flux.state(model.gnn |> Flux.cpu)
    jldsave(filepath; model_state, meta)
    @info "Model successfully saved to $filepath"
end

"""
    load_model!(model::DiffusionModel, filepath::String; expected_meta=nothing, strict_keys=nothing)

Loads the state of the GNN model from a JLD2 file into the provided model object. With `expected_meta`, the
stored training fingerprint is compared with it: a mismatch in one of `strict_keys` (default: all keys) is an
error, any other mismatch a warning. Files saved without a fingerprint (older runs) are loaded with a warning.

Parameters:
    - 'model': The model to be loaded.
    - 'filepath': The filepath where the model is located.
    - 'expected_meta': The fingerprint of the current configuration, or `nothing` to skip the check.
    - 'strict_keys': The keys whose mismatch is an error.
"""
function load_model!(model::DiffusionModel, filepath::String;
                     expected_meta::Union{Nothing,Dict{String,String}}=nothing,
                     strict_keys::Union{Nothing,AbstractVector{String}}=nothing)
    if !isfile(filepath)
        error("Model file not found: $filepath")
    end

    model_state, stored = jldopen(filepath, "r") do f
        f["model_state"], (haskey(f, "meta") ? f["meta"] : nothing)
    end

    if !isnothing(expected_meta)
        if isnothing(stored) || isempty(stored)
            @warn "Model $filepath has no stored training fingerprint (saved before fingerprints were added); " *
                  "cannot verify that it was trained with the current configuration."
        else
            strict = isnothing(strict_keys) ? collect(keys(expected_meta)) : strict_keys
            diff = [k for (k, v) in expected_meta if get(stored, k, "<missing>") != v]
            detail = join(["$k: stored '$(get(stored, k, "<missing>"))', current '$(expected_meta[k])'" for k in diff], "; ")
            if any(in(strict), diff)
                error("Model $filepath was trained with a different configuration ($detail). " *
                      "Move or delete the file, or change the train_run_id.")
            elseif !isempty(diff)
                @warn "Model $filepath: training settings differ from the current ones ($detail); loading anyway."
            end
        end
    end

    Flux.loadmodel!(model.gnn, model_state)
    @info "Model successfully loaded from $filepath"

    return model
end
