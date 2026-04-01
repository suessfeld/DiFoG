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