using TestItems
using Thesis

@testitem "Generate a small instance collection for MISP" setup=[GenerationSetup] tags=[:job] begin
    using .GenerationSetup.InstanceUtils: generate_collection
    using .GenerationSetup: solve_misp_instance

    collection_name = "misp_small_instances"
    num_instances = 1000

    min_vertices = 20
    max_vertices = 50
    min_edges = 20
    max_edges = 50

    generate_collection(collection_name, num_instances, min_vertices, max_vertices, min_edges, max_edges, solve_misp_instance)
end

@testitem "Generate a medium instance collection for MISP" setup=[GenerationSetup] tags=[:job] begin
    using .GenerationSetup.InstanceUtils: generate_collection
    using .GenerationSetup: solve_misp_instance

    collection_name = "misp_medium_instances"
    num_instances = 1000

    min_vertices = 50
    max_vertices = 100
    min_edges = 200
    max_edges = 300

    generate_collection(collection_name, num_instances, min_vertices, max_vertices, min_edges, max_edges, solve_misp_instance)
end

@testitem "Generate a large instances collection for MISP" setup=[GenerationSetup] tags=[:job] begin
    using .GenerationSetup.InstanceUtils: generate_collection
    using .GenerationSetup: solve_misp_instance

    collection_name = "misp_large_instances"
    num_instances = 1000

    min_vertices = 100
    max_vertices = 200
    min_edges = 300
    max_edges = 400

    generate_collection(collection_name, num_instances, min_vertices, max_vertices, min_edges, max_edges, solve_misp_instance)
end

@testitem "Generate a small instance collection for MAXDEG" setup=[GenerationSetup] tags=[:job] begin
    using .GenerationSetup.InstanceUtils: generate_collection
    using .GenerationSetup: solve_max_degree_instance

    collection_name = "maxdeg_small_instances"
    num_instances = 1000

    min_vertices = 20
    max_vertices = 50
    min_edges = 20
    max_edges = 50

    generate_collection(collection_name, num_instances, min_vertices, max_vertices, min_edges, max_edges, solve_max_degree_instance)
end

@testitem "Generate a medium instance collection for MAXDEG" setup=[GenerationSetup] tags=[:job] begin
    using .GenerationSetup.InstanceUtils: generate_collection
    using .GenerationSetup: solve_max_degree_instance

    collection_name = "maxdeg_medium_instances"
    num_instances = 1000

    min_vertices = 50
    max_vertices = 100
    min_edges = 200
    max_edges = 300

    generate_collection(collection_name, num_instances, min_vertices, max_vertices, min_edges, max_edges, solve_max_degree_instance)
end

@testitem "Generate a large instance collection for MAXDEG" setup=[GenerationSetup] tags=[:job] begin
    using .GenerationSetup.InstanceUtils: generate_collection
    using .GenerationSetup: solve_max_degree_instance

    collection_name = "maxdeg_large_instances"
    num_instances = 50000

    min_vertices = 100
    max_vertices = 200
    min_edges = 400
    max_edges = 500

    generate_collection(collection_name, num_instances, min_vertices, max_vertices, min_edges, max_edges, solve_max_degree_instance)
end

@testmodule GenerationSetup begin
    using DotEnv
    using Thesis
    using Graphs
    using SimpleGraphAlgorithms
    using ChooseOptimizer
    using GLPK

    env_path = joinpath(pkgdir(Thesis), "src", ".env")
    if isfile(env_path)
        DotEnv.load!(env_path)
    else
        @warn "No .env file found at $env_path"
    end

    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "instance_utils.jl"))
    using .InstanceUtils

    # Include the problem-specific solvers directly into TestSetup scope
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "problems", "misp_utils.jl"))
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "problems", "max_degree_utils.jl"))
end