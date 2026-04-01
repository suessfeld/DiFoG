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
    half = div(embedding_dim, 2)

    freqs = Zygote.ignore_derivatives() do
        f = Float32.(0:half-1) ./ Float32(half)
        exp.(-log(Float32(max_period)) .* f) |> Flux.gpu
    end

    t = Float32.(timesteps) |> Flux.gpu   # (T,)
    
    args = reshape(t, :, 1) * reshape(freqs, 1, :)   # (T, half)

    emb = hcat(cos.(args), sin.(args))    # (T, embedding_dim)
    emb = permutedims(emb)                # (embedding_dim, T)

    if embedding_dim % 2 == 1
        emb = vcat(emb, CUDA.zeros(Float32, 1, size(emb, 2)))
    end

    return emb
end