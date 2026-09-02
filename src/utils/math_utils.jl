using Flux
using CUDA

#GR TODO Eine Adjacency Matrix kann directer in Graphs.jl über die Funktion adjacency_Matrix erhalten werden. Ich sehe aber eigentlich nicht, wofür wir dies benötigen sollten. Üblicherweise arbeiten wir hier mit dünnen Graphen und da ist der Umweg über eine Adjazenzmatrix fast nie die effizienteste Lösung.

"""
    adjacency_list_to_matrix(adj_list::Vector{Vector{Int}})

Converts an adjacency list to a 2xE edge matrix.

Parameters:
    - 'adj_list': The adjacency list to transform.
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