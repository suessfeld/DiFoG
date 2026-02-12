module DataUtils

using Random
using SimpleGraphs

export generate_graph, load_graph, load_instance

"""
Generate a random graph to the file with the defined size and label
"""
function generate_graph(filename::String; num_nodes=10, num_edges=20, label=1)
    if !isdir(dir_path)
        mkpath(dir_path)
        println("Created directory: $dir_path")
    end

    possible_edges = Tuple{Int, Int}[]
    for i in 1:num_nodes
        for j in (i+1):num_nodes
            push!(possible_edges, (i, j))
        end
    end

    max_edges = length(possible_edges)
    if num_edges > max_edges
        @warn "Requested $num_edges edges, but only $max_edges are possible for $num_nodes nodes. Generating full graph."
        num_edges = max_edges
    end

    selected_edges = shuffle(possible_edges)[1:num_edges]
    
    sort!(selected_edges)

    open(filename, "w") do io
        println(io, num_nodes)
        println(io, num_edges)
        println(io, label)
        
        for (u, v) in selected_edges
            println(io, "$u $v")
        end
    end
end

"""
Load a graph from a file in the specified format and return it as an UndirectedGraph.
"""
function load_graph(filename::String)
    lines = readlines(filename)

    num_nodes = parse(Int, lines[1]) # line 1
    num_edges_declared = parse(Int, lines[2]) # line 2
    label = parse(Int, lines[3]) # line 3
    
    G = UndirectedGraph{Int}()

    for i in 1:num_nodes
        add_vertex!(G, i)
    end

    for i in 4:length(lines)

        parts = split(line)
        u = parse(Int, parts[1])
        v = parse(Int, parts[2])
        add_edge!(G, u, v)

    end
    return G
end

load_instance(name::String) = load_graph(joinpath(ENV["TEST_CASES_PATH"], "$name.txt"))

end