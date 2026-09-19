# src/graphs/basic.jl
#
# A basic random graph generator that creates graph instances within specified node and edge count ranges.

using Graphs
using Random

export BasicGraphGenerator

"""
    BasicGraphGenerator <: GraphGenerator
    
A basic graph generator that generates according to the defined ranged of nodes and edges.
"""
struct BasicGraphGenerator <: GraphGenerator
    min_nodes::Int
    max_nodes::Int
    min_edges::Int
    max_edges::Int
end

function BasicGraphGenerator(; min_nodes::Int=50, max_nodes::Int=100, min_edges::Int=200, max_edges::Int=300)
    return BasicGraphGenerator(min_nodes, max_nodes, min_edges, max_edges)
end

function (gen::BasicGraphGenerator)(rng::AbstractRNG=Random.default_rng())
    num_nodes = rand(rng, gen.min_nodes:gen.max_nodes)

    # Safely clamp the edge bounds based on the randomly rolled node count
    min_e = max(gen.min_edges, num_nodes - 1)
    max_e = min(gen.max_edges, div(num_nodes * (num_nodes - 1), 2))
    min_e = min(min_e, max_e)

    num_edges = rand(rng, min_e:max_e)

    selected_edges = Tuple{Int, Int}[]
    connected_nodes = [1]
    unconnected_nodes = shuffle(rng, 2:num_nodes)

    for u in unconnected_nodes
        v = rand(rng, connected_nodes)
        push!(selected_edges, (min(u, v), max(u, v)))
        push!(connected_nodes, u)
    end

    tree_edges = Set(selected_edges)
    possible_edges = Tuple{Int, Int}[]

    for i in 1:num_nodes
        for j in (i+1):num_nodes
            edge = (i, j)
            if !(edge in tree_edges)
                push!(possible_edges, edge)
            end
        end
    end

    remaining_edges_needed = num_edges - length(selected_edges)
    if remaining_edges_needed > 0
        append!(selected_edges, shuffle(rng, possible_edges)[1:remaining_edges_needed])
    end

    g = SimpleGraph(num_nodes)
    for (u, v) in selected_edges
        add_edge!(g, u, v)
    end

    return g
end