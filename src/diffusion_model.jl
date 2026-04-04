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
    - 'g': a graph(batch)
    - 'x': a feature matrix
"""

function (model::DiffusionModel)(g::GNNGraph, x_t::AbstractMatrix)
    return model.gnn(g, x_t)
end

"""
    (model::DiffusionModel)(x::AbstractArray, t::AbstractVector{Int}, g::GNNGraph)

Defines a forward pass (callable for Flux) of the network

Parameters:
    - 'x_t': current state of the solution
    - 'g': a graph(batch)
"""
_embed_timesteps(t_emb, ::Nothing, x_t) =
    repeat(t_emb, 1, size(x_t, 2))

_embed_timesteps(t_emb, node_indices::AbstractVector, x_t) =
    t_emb[:, node_indices]

function (model::DiffusionModel)(x_t::AbstractArray, t::AbstractVector{<:Integer}, g::GNNGraph)
    x_dim    = size(x_t, 1)
    time_dim = model.gnn.config.in_dim - x_dim

    t_emb = Thesis.get_timestep_embedding(t, time_dim)
    t_emb_nodes = _embed_timesteps(t_emb, g.graph_indicator, x_t)
    x_input = vcat(t_emb_nodes, x_t)
    result = model.gnn(g, x_input)
    return result
end

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
                g_gpu = g_batch |> Flux.gpu
                x_start = g_gpu.ndata.x 
                x_t, t_dev, target = corrupt_batch(model.method, x_start, g_gpu)
                
                val, grads = compute_grad_barrier(model.gnn, model.method, x_t, t_dev, g_gpu, target)
                
                Flux.update!(optim, model.gnn, grads[1])
            end
            
            avg_time_per_instance = batch_time / batch_size
            
            @debug "Batch $(num_batches + 1) [$(batch_size*num_batches) instances total] | Total: $(round(batch_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

            epoch_loss += val
            num_batches += 1
        end
        
        avg_loss = epoch_loss / max(1, num_batches)
        push!(loss_history, avg_loss)
        @info "Epoch $epoch | Avg Loss: $avg_loss"
        
        GC.gc(true)
        CUDA.reclaim()
    end
    
    return loss_history
end

function infer(model::DiffusionModel, g_batch::GNNGraph)
    g_gpu = g_batch |> Flux.gpu
    shape = (1, g_gpu.num_nodes)
    heatmap_gpu = Thesis.sample_heatmap(model.method, model, shape, g_gpu)
    return heatmap_gpu |> Flux.cpu
end


function compute_grad_barrier(gnn::GNN, method::Diffusion, x_t::AbstractArray, t_dev::AbstractVector, g_gpu::GNNGraph, target::AbstractArray)
    x_dim = size(x_t, 1)
    time_dim = gnn.config.in_dim - x_dim
    t_emb = Thesis.get_timestep_embedding(t_dev, time_dim)
    t_emb_nodes = isnothing(g_gpu.graph_indicator) ?
        repeat(t_emb, 1, size(x_t, 2)) :
        t_emb[:, g_gpu.graph_indicator]

    x_input = vcat(t_emb_nodes, x_t)  # all outside the closure

    val, grads = Flux.withgradient(gnn) do gnn_inner
        preds = gnn_inner(g_gpu, x_input)
        compute_loss_value(method, preds, target)
    end
    return val, grads
end