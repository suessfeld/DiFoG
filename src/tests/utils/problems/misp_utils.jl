
export solve_misp_instance

function solve_misp_instance(instance_path::String, solution_path::String)
    set_solver(GLPK)

    graph_original = load_instance(instance_path)
    graph_transformed = convert_to_simplegraph(graph_original)
    mis_set = max_indep_set(graph_transformed)

    content = format_solution(nv(graph_original), mis_set)
    open(solution_path, "w") do io
        @info "solved instance $solution_path"
        write(io, content)
    end
end