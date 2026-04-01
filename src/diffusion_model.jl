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
function (model::DiffusionModel)(x_t::AbstractArray, t::AbstractVector{Int}, g::GNNGraph)
   
    x_dim = size(x_t, 1)
    time_dim = model.gnn.config.in_dim - x_dim

    # get time embeddings
    t_emb = Thesis.get_timestep_embedding(t, time_dim)
    
    # for batch mapping
    node_indices = g.graph_indicator
    
    if isnothing(node_indices) # single graph case
        num_nodes = size(x_t, 2)
        t_emb_nodes = repeat(t_emb, 1, num_nodes)
    else # batched case
        t_emb_nodes = t_emb[:, node_indices]
    end
    
    x_input = vcat(x_t, t_emb_nodes)
    return model.gnn(g, x_input)
end

function train_diffusion_model!(model::DiffusionModel, optim, train_loader; epochs::Int=5)
    loss_history = Float32[]
    
    for epoch in 1:epochs
        epoch_loss = 0.0f0
        num_batches = 0
        
        # DataLoader yields batches. For GNNGraphs, they are merged into disjoint graphs.
        for (g_batch, names_batch) in train_loader
            # 1. Just-in-time transfer to GPU
            g_gpu = g_batch |> Flux.gpu
            x_start = g_gpu.ndata.x 
            
            # 2. Forward pass & gradient calculation
            val, grads = Flux.withgradient(model) do m
                Thesis.compute_loss(m.method, m, x_start, g_gpu)
            end
            
            # 3. Update weights
            Flux.update!(optim, model, grads[1])
            
            epoch_loss += val
            num_batches += 1
        end
        
        avg_loss = epoch_loss / max(1, num_batches)
        push!(loss_history, avg_loss)
        @info "Epoch $epoch | Avg Loss: $avg_loss"
        
        # Proactive garbage collection for VRAM stability across large epoch boundaries
        GC.gc(true)
        CUDA.reclaim()
    end
    
    return loss_history
end

function infer(model::DiffusionModel, graphs::AbstractVector{<:GNNGraph})
    heatmaps = []
    
    for g in graphs
        g_gpu = g |> Flux.gpu
        num_nodes = g_gpu.num_nodes
        shape = (1, num_nodes)
        
        heatmap = Thesis.sample_heatmap(model.method, model, shape, g_gpu)
        
        # Return the result back to the CPU immediately to free VRAM
        push!(heatmaps, heatmap |> Flux.cpu)
    end
    
    return heatmaps
end

"""
    infer(model::DiffusionModel, g::GNNGraph)

Convenience wrapper for running inference on a single graph.
"""
function infer(model::DiffusionModel, g::GNNGraph)
    return infer(model, [g])[1]
end