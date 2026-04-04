# src/tests/utils/data_utils.jl
module DataUtils

using Flux
using GraphNeuralNetworks
using Random
using Graphs: nv
using Thesis
using ..InstanceUtils: load_instance

export load_collection, split_dataset
export LazyGraphDataset, load_lazy_collection, split_lazy_dataset

struct LazyGraphDataset
    instances_dir::String
    solutions_dir::String
    valid_files::Vector{String}
end

Base.length(d::LazyGraphDataset) = length(d.valid_files)

function normalize_graph(g::GNNGraph)
    s, t = edge_index(g)  # always returns plain Int vectors
    return GNNGraph(
        Int32.(s), Int32.(t);
        num_nodes = g.num_nodes,
        ndata     = g.ndata,   # preserves :x
        edata     = g.edata,
        gdata     = g.gdata,
    )
end

function Base.getindex(d::LazyGraphDataset, i::Int)
    file = d.valid_files[i]
    inst_path = joinpath(d.instances_dir, file)
    sol_path = joinpath(d.solutions_dir, replace(file, ".txt" => "_solution.txt"))
    
    raw_g = load_instance(inst_path)
    lines = readlines(sol_path)
    
    x_0 = zeros(Float32, 1, nv(raw_g))
    if length(lines) >= 3
        mask_str = split(lines[3])
        mask = parse.(Float32, mask_str)
        x_0 = reshape(mask, 1, :)
    end
    
    gnn_g = GNNGraph(raw_g, ndata=(; x=x_0))
    return (normalize_graph(gnn_g), file)  # ← only change to this function
end

function Base.getindex(d::LazyGraphDataset, idxs::AbstractVector{<:Integer})
    tasks = map(idxs) do i
        Threads.@spawn d[i]
    end
    return fetch.(tasks)
end

function load_lazy_collection(instance_collection::String)
    collection_dir = joinpath(pkgdir(Thesis), ENV["INSTANCES_PATH"], instance_collection)
    instances_dir = joinpath(collection_dir, "instances")
    solutions_dir = joinpath(collection_dir, "solutions")
    inferences_dir = joinpath(collection_dir, "inferences")

    if !isdir(inferences_dir)
        mkpath(inferences_dir)
    end
    
    if !isdir(solutions_dir)
        @warn "Solutions directory not found. Please run the Solver Job first."
        return LazyGraphDataset(instances_dir, solutions_dir, String[]), inferences_dir
    end

    files = sort(readdir(instances_dir))
    valid_files = String[]
    for file in files
        if endswith(file, ".txt")
            sol_path = joinpath(solutions_dir, replace(file, ".txt" => "_solution.txt"))
            if isfile(sol_path)
                push!(valid_files, file)
            end
        end
    end
    
    return LazyGraphDataset(instances_dir, solutions_dir, valid_files), inferences_dir
end

function split_lazy_dataset(dataset::LazyGraphDataset, train_ratio::Float64=0.9)
    total = length(dataset)
    shuffled_files = shuffle(dataset.valid_files)
    
    train_size = max(1, round(Int, train_ratio * total))
    
    train_files = shuffled_files[1:train_size]
    test_files = shuffled_files[train_size+1:end]
    
    train_dataset = LazyGraphDataset(dataset.instances_dir, dataset.solutions_dir, train_files)
    test_dataset = LazyGraphDataset(dataset.instances_dir, dataset.solutions_dir, test_files)
    
    return train_dataset, test_dataset
end

function load_collection(instance_collection::String)
    collection_dir = joinpath(pkgdir(Thesis), ENV["INSTANCES_PATH"], instance_collection)
    instances_dir = joinpath(collection_dir, "instances")
    solutions_dir = joinpath(collection_dir, "solutions")
    inferences_dir = joinpath(collection_dir, "inferences")

    if !isdir(inferences_dir)
        mkpath(inferences_dir)
    end
    
    all_pairs = Tuple{GNNGraph, String}[] 
    
    if !isdir(solutions_dir)
        @warn "Solutions directory not found. Please run the Solver Job first."
        return all_pairs, inferences_dir
    end

    files = sort(readdir(instances_dir))
    for file in files
        if endswith(file, ".txt")
            inst_path = joinpath(instances_dir, file)
            sol_path = joinpath(solutions_dir, replace(file, ".txt" => "_solution.txt"))
            
            if isfile(sol_path)
                raw_g = load_instance(inst_path)
                lines = readlines(sol_path)
                if length(lines) >= 3
                    mask_str = split(lines[3])
                    mask = parse.(Float32, mask_str)
                    x_0 = reshape(mask, 1, :)
                    gnn_g = GNNGraph(raw_g, ndata=(; x=x_0))
                    push!(all_pairs, (gnn_g, file))
                end
            end
        end
    end
    return all_pairs, inferences_dir
end

function split_dataset(all_pairs::Vector{Tuple{GNNGraph, String}}, train_ratio::Float64=0.9)
    total_instances = length(all_pairs)
    test_size = max(1, round(Int, (1.0 - train_ratio) * total_instances))
    train_size = total_instances - test_size
    shuffled_pairs = shuffle(all_pairs)
    return shuffled_pairs[1:train_size], shuffled_pairs[train_size+1:end]
end

end