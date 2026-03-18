export solve_max_degree_instance

"""
    solve_max_degree_instance(instance_path::String, solution_path::String)

Loads an instance, finds all nodes holding the maximum degree, and writes the 
solution in a 0/1 mask format identical to the MISP formulation.
"""
function solve_max_degree_instance(instance_path::String, solution_path::String)
    # Load the graph using InstanceUtils
    graph_original = load_instance(instance_path)
    
    # Get all degrees and find the maximum degree
    degs = degree(graph_original)
    max_deg = maximum(degs)
    
    # Find the indices of all nodes that have this maximum degree
    max_deg_nodes = findall(x -> x == max_deg, degs)

    # Format the solution exactly like MISP (requires format_solution from InstanceUtils)
    content = format_solution(nv(graph_original), max_deg_nodes)
    
    open(solution_path, "w") do io
        @info "solved max degree instance $solution_path"
        write(io, content)
    end
end