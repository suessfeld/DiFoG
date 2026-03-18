using Flux
using Zygote

"""
    adjacency_list_to_matrix(adj_list::Vector{Vector{Int}})

Converts an adjacency list to a 2xE edge matrix.
"""
function adjacency_list_to_matrix(adj_list::Vector{Vector{Int}})
    src = Int[]
    dst = Int[]
    for (i, neighbors) in enumerate(adj_list)
        for n in neighbors
            push!(src, i)
            push!(dst, n)
        end
    end
    return hcat(src, dst)' # Transpose to get 2xE
end

"""
    get_timestep_embedding(timesteps, embedding_dim, max_period=10000)

Generates sinusoidal time embeddings.
Args:
    timesteps: Vector of integers (T,).
    embedding_dim: Output dimension (D,).
Returns:
    Embeddings of shape (D, T).
"""
function get_timestep_embedding(timesteps::AbstractVector{Int}, embedding_dim::Int, max_period=10000)
    # Use Zygote.ignore to prevent the AD from trying to differentiate 
    # the 'range' constructor and integer arithmetic, which causes the StepRangeLen error.
    return Zygote.ignore() do
        half = div(embedding_dim, 2)
        freqs = exp.(-log(max_period) .* range(0, stop=1, length=half))
        args = Float32.(timesteps) .* freqs'
        embedding = vcat(cos.(args'), sin.(args'))
        
        # If odd dimension, pad with zero (though typically even is used)
        if embedding_dim % 2 == 1
            embedding = vcat(embedding, zeros(Float32, 1, length(timesteps)))
        end
        return embedding
    end
end