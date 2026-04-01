module InstanceUtils

using Random
using Graphs
using Thesis

import SimpleGraphs
using SimpleGraphAlgorithms
using ChooseOptimizer
using GLPK

export generate_collection, load_instance, format_solution, convert_to_simplegraph

function generate_collection(collection_name::String, num_instances::Int, min_vertices::Int, max_vertices::Int, min_edges::Int, max_edges::Int, solver::Function)
    output_dir = joinpath(pkgdir(Thesis), ENV["INSTANCES_PATH"], collection_name)
    instances_dir = joinpath(output_dir, "instances")
    solutions_dir = joinpath(output_dir, "solutions")
    inferences_dir = joinpath(output_dir, "inferences")

    if !isdir(output_dir)
        mkpath(output_dir)
    end

    if !isdir(instances_dir)
        mkpath(instances_dir)
    end

    if !isdir(solutions_dir)
        mkpath(solutions_dir)
    end

    if !isdir(inferences_dir)
        mkpath(inferences_dir)
    end

    generated_filenames = String[]

    @info "generating instances under $instances_dir."
    
    for i in 1:num_instances
        v = rand(min_vertices:max_vertices)
        e = rand(min_edges:max_edges)
        
        filename = "$(collection_name)_$(i)_$(v)_$(e).txt"
        full_path = joinpath(instances_dir, filename)
        
        generate_instance(full_path; num_nodes=v, num_edges=e)
        push!(generated_filenames, filename)
    end

    @info "solving instances."

    for filename in generated_filenames
        solver(
            joinpath(instances_dir, filename),
            joinpath(solutions_dir, replace(filename, ".txt" => "_solution.txt"))
        )
    end
end

function generate_instance(filename::String; num_nodes=10, num_edges=20)
    @info "generate instance $filename"
    if num_edges < num_nodes - 1
        @warn "num_edges ($num_edges) is less than num_nodes - 1 ($(num_nodes - 1)). Clamping to $(num_nodes - 1) to ensure connectivity."
        num_edges = num_nodes - 1
    end

    max_edges = div(num_nodes * (num_nodes - 1), 2)
    if num_edges > max_edges
        num_edges = max_edges
    end

    selected_edges = Tuple{Int, Int}[]
    
    connected_nodes = [1]
    unconnected_nodes = shuffle(2:num_nodes)
    
    for u in unconnected_nodes
        v = rand(connected_nodes)
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
        append!(selected_edges, shuffle(possible_edges)[1:remaining_edges_needed])
    end
    
    sort!(selected_edges)

    open(filename, "w") do io
        println(io, num_nodes)
        println(io, num_edges)
        
        for (u, v) in selected_edges
            println(io, "$u $v")
        end
    end
end

function load_instance(filename::String)
    if !isfile(filename)
        error("File not found: $filename")
    end

    lines = readlines(filename)
    num_nodes = parse(Int, lines[1])
    
    G = SimpleGraph(num_nodes)

    for i in 3:length(lines)
        isempty(lines[i]) && continue
        parts = split(lines[i])
        u = parse(Int, parts[1])
        v = parse(Int, parts[2])
        add_edge!(G, u, v)
    end
    return G
end

function convert_to_simplegraph(g_graphs)
    sg = SimpleGraphs.UndirectedGraph()

    for v in 1:nv(g_graphs)
        SimpleGraphs.add!(sg, v)
    end

    for e in edges(g_graphs)
        SimpleGraphs.add!(sg, src(e), dst(e))
    end
    
    return sg
end

function format_solution(num_nodes::Int, mis_set)
    sorted_set = sort(collect(mis_set))
    size_mis = length(sorted_set)
    mask = zeros(Int, num_nodes)
    
    for idx in sorted_set
        if idx <= num_nodes
            mask[idx] = 1
        end
    end
    
    l1 = string(size_mis)
    l2 = join(sorted_set, " ")
    l3 = join(mask, " ")
    
    return "$l1\n$l2\n$l3"
end

end