export DiffusionModel

struct DiffusionModel{A<:GNN, P<:Diffusion}
    gnn::A
    process::P
end

Flux.@functor DiffusionModel

function (model::DiffusionModel)(g::GNNGraph, x::AbstractMatrix)
    return model.gnn(g, x)
end

"""
    (wrapper::GraphDiffusionModel)(x::AbstractArray, t::AbstractVector{Int}, g::GNNGraph)

Forward pass compatible with diffusion compute_loss. 
Injects time embeddings into node features.
"""
function (model::DiffusionModel)(x::AbstractArray, t::AbstractVector{Int}, g::GNNGraph)
    # x is (Features, TotalNodes)
    # t is (NumGraphs,) i.e., batch size
    
    # 1. Calculate Time Embeddings
    x_dim = size(x, 1)
    time_dim = model.gnn.config.in_dim - x_dim
    
    if time_dim <= 0
        error("Model input dimension ($(model.gnn.config.in_dim)) must be larger than data dimension ($x_dim) to accommodate time embeddings.")
    end

    # t_emb: (TimeDim, NumGraphs)
    t_emb = Thesis.get_timestep_embedding(t, time_dim)
    
    # 2. Broadcast to Nodes
    # g.graph_indicator maps node -> graph index. 
    # For single graphs, it might be nothing.
    node_indices = g.graph_indicator
    
    if isnothing(node_indices)
        # Fallback for single graph instances
        # We assume all nodes belong to the first (and only) graph in the batch.
        # We verify t is scalar-like (length 1) to prevent semantic errors.
        if length(t) != 1
            # If we have multiple timestamps but no way to assign them to nodes, it's an error.
            error("Ambiguous time assignment: Batch size is $(length(t)) but graph_indicator is missing.")
        end
        
        # Broadcast the single time embedding to all nodes
        # t_emb is (TimeDim, 1). We repeat it to (TimeDim, NumNodes).
        num_nodes = size(x, 2)
        t_emb_nodes = repeat(t_emb, 1, num_nodes)
    else
        # Standard batched case: Map columns of t_emb to nodes
        t_emb_nodes = t_emb[:, node_indices]
    end
    
    # 3. Concatenate (Features, Nodes)
    x_input = vcat(x, t_emb_nodes)
    
    # 4. GNN Forward
    return model.gnn(g, x_input)
end