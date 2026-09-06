# src/utils/model_utils.jl
#
# Utilities for safely saving and loading trained GNN model states to and from disk using JLD2.

using Flux
using JLD2

export save_model, load_model!

"""
    save_model(model::DiffusionModel, filepath::String)

Saves the state of the GNN model to a JLD2 file. Moves the model to CPU before saving.

Parameters:
    - 'model': The model to be saved.
    - 'filepath': The filepath where the model should be saved.
"""
function save_model(model::DiffusionModel, filepath::String)
    model_state = Flux.state(model.gnn |> Flux.cpu)
    jldsave(filepath; model_state)
    @info "Model successfully saved to $filepath"
end

"""
    load_model!(model::DiffusionModel, filepath::String)

Loads the state of the GNN model from a JLD2 file into the provided model object.

Parameters:
    - 'model': The model to be loaded.
    - 'filepath': The filepath where the model is located.
"""
function load_model!(model::DiffusionModel, filepath::String)
    if !isfile(filepath)
        error("Model file not found: $filepath")
    end
    
    file_data = jldopen(filepath, "r")
    model_state = file_data["model_state"]
    close(file_data)
    
    Flux.loadmodel!(model.gnn, model_state)
    @info "Model successfully loaded from $filepath"
    
    return model
end