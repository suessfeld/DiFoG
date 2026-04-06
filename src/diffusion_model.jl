using Profile

export DiffusionModel

struct DiffusionModel{G<:GNN, P<:Diffusion}
    gnn::G
    method::P
end

Flux.@layer DiffusionModel

"""
    (model::DiffusionModel)(g::GNNGraph, x::AbstractMatrix)

Constructor

Parameters:
    -'g': a graph(batch)
    -'x': a feature matrix
"""

function (model::DiffusionModel)(g::GNNGraph, x_t::AbstractMatrix)
    return model.gnn(g, x_t)
end


_embed_timesteps(t_emb, ::Nothing, x_t) =
    repeat(t_emb, 1, size(x_t, 2))

_embed_timesteps(t_emb, node_indices::AbstractVector, x_t) =
    t_emb[:, node_indices]

"""
    (model::DiffusionModel)(x::AbstractArray, t::AbstractVector{Int}, g::GNNGraph)

Defines a forward pass (callable for Flux) of the network

Parameters:
    -'x_t': current state of the solution
    -'g': a graph(batch)
"""
function (model::DiffusionModel)(x_t::AbstractArray, t::AbstractVector{<:Integer}, g::GNNGraph)
    x_dim    = size(x_t, 1)
    time_dim = model.gnn.config.in_dim - x_dim

    t_emb = Thesis.get_timestep_embedding(t, time_dim)
    t_emb_nodes = _embed_timesteps(t_emb, g.graph_indicator, x_t)
    x_input = vcat(t_emb_nodes, x_t)
    result = model.gnn(g, x_input)
    return result
end

"""
    train_diffusion_model!(model::DiffusionModel, optim, data_loader; epochs::Int=5)

Main training loop of a diffusion model.

Parameters:
    -'model': the model to train
    -'optim': the optimizer
    -'data_loader': dataloader which loads training data lazily
    -'epochs': the epochs to train for
"""
function train_diffusion_model!(model::DiffusionModel, optim, data_loader; epochs::Int=5)
    loss_history = Float32[]
    @info "Starting training"
    
    for epoch in 1:epochs
        epoch_loss = 0.0f0
        num_batches = 1
        
        prefetch_channel = Channel(3, spawn=true) do ch
            for batch in data_loader
                put!(ch, batch)
            end
        end
        
        for (g_batch, names_batch) in prefetch_channel
            batch_size = length(names_batch)
            
            batch_time = @elapsed begin
                g_batch = g_batch |> Flux.gpu # move entire batch to gpu
                x_start = g_batch.ndata.x 
                x_t, timesteps, x_0 = corrupt_batch(model.method, x_start, g_batch)
                
                loss, grads = compute_loss_and_gradients(model.gnn, model.method, x_t, timesteps, g_batch, x_0)
                
                Flux.update!(optim, model.gnn, grads[1])
            end
            
            avg_time_per_instance = batch_time / batch_size
            
            @info "Batch $(num_batches + 1) [$(batch_size*num_batches) instances total] | Total: $(round(batch_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

            epoch_loss += loss
            num_batches += 1
        end
        
        avg_loss = epoch_loss / max(1, num_batches)
        push!(loss_history, avg_loss)
        @info "Epoch $epoch | Avg Loss: $avg_loss"
    end
    
    return loss_history
end

"""
    infer(model::DiffusionModel, g_batch::GNNGraph)

Inference call to the trained model. Moves the entire batch to the GPU and processes the inference there.

Parameters:
    -'model': the model which will be infered
    -'batch': the batch of graphs which will be passed for inference
"""
function infer(model::DiffusionModel, batch::GNNGraph)
    g_gpu = batch |> Flux.gpu
    shape = (1, g_gpu.num_nodes)
    heatmap_gpu = Thesis.sample_heatmap(model.method, model, shape, g_gpu)
    return heatmap_gpu |> Flux.cpu
end

"""
    compute_loss_and_gradients

Consists of one forward and one backward pass through the network. Returns the gradient and the loss for weight adjustments.

Parameters:
    -'gnn': the GNN of the model
    -'method': the diffusion method
    -'x_t': solution state at timestep t
    -'timesteps': vector containing the diffusion timesteps for each graph in the batch
    -'batch': the batch of graphs, already on the GPU
    -'x_0': the optimal solution for loss computation
"""
function compute_loss_and_gradients(gnn::GNN, method::Diffusion, x_t::AbstractArray, timesteps::AbstractVector, batch::GNNGraph, x_0::AbstractArray)
    x_dim = size(x_t, 1)
    time_dim = gnn.config.in_dim - x_dim
    t_emb = Thesis.get_timestep_embedding(timesteps, time_dim)
    t_emb_nodes = isnothing(batch.graph_indicator) ?
        repeat(t_emb, 1, size(x_t, 2)) :
        t_emb[:, batch.graph_indicator]

    x_input = vcat(t_emb_nodes, x_t)

    loss, grads = Flux.withgradient(gnn) do gnn_inner # IMPORTANT TO BOX LIKE THIS FOR ZYGOTE! Otherwise each training call the gradient will be recalculated!
        preds = gnn_inner(batch, x_input)
        compute_loss_value(method, preds, x_0)
    end
    return loss, grads
end