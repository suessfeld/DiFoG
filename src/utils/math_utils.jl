using Flux
using CUDA

"""
    adjacency_list_to_matrix(adj_list::Vector{Vector{Int}})

Converts an adjacency list to a 2xE edge matrix.

Parameters:
    -'adj_list': the adjacency list to transform
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

Parameters:
    -'timesteps': vector of integers (T,)
    -'embedding_dim': output dimension (D,)
    -'max_period': max period
"""
function get_timestep_embedding(timesteps::AbstractVector{<:Integer}, embedding_dim::Int, max_period=10000)
    half = div(embedding_dim, 2)
    freqs = exp.(-log(Float32(max_period)) .* Float32.(0:half-1) ./ Float32(half)) |> Flux.gpu
    t = Float32.(timesteps)
    args = reshape(t, :, 1) .* reshape(freqs, 1, :)
    emb = permutedims(hcat(cos.(args), sin.(args)))
    if embedding_dim % 2 == 1
        emb = vcat(emb, CUDA.zeros(Float32, 1, size(emb, 2)))
    end
    return emb
end