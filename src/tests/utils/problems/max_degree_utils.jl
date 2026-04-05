export solve_max_degree_instance

function solve_max_degree_instance(instance_path::String, solution_path::String)
    graph_original = load_instance(instance_path)
    
    # Get all degrees and find the maximum degree
    degs = degree(graph_original)
    max_deg = maximum(degs)
    max_deg_nodes = findall(x -> x == max_deg, degs)

    # make sure the formatting is consistent
    content = format_solution(nv(graph_original), max_deg_nodes)
    
    open(solution_path, "w") do io
        @info "solved max degree instance $solution_path"
        write(io, content)
    end
end