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