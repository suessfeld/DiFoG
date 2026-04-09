using Flux
using JLD2

export save_model, load_model!

"""
    save_model(model::DiffusionModel, filepath::String)

Saves the state of the GNN model to a JLD2 file. Moves the model to CPU before saving.
"""
function save_model(model::DiffusionModel, filepath::String)
    model_state = Flux.state(model.gnn |> Flux.cpu)
    jldsave(filepath; model_state)
    @info "Model successfully saved to $filepath"
end

"""
    load_model!(model::DiffusionModel, filepath::String)

Loads the state of the GNN model from a JLD2 file into the provided model object.
"""
function load_model!(model::DiffusionModel, filepath::String)
    if !isfile(filepath)
        error("Model file not found: $filepath")
    end
    
    file_data = jldopen(filepath, "r")
    model_state = file_data["model_state"]
    close(file_data)
    
    # Load the parameters back into the model's GNN
    Flux.loadmodel!(model.gnn, model_state)
    @info "Model successfully loaded from $filepath"
    
    return model
end