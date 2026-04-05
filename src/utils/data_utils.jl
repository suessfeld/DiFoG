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
    files::Vector{String}
end

# define length
Base.length(d::LazyGraphDataset) = length(d.files)

"""
    function normalize_graph(g::GNNGraph)

Normalizes an input graph so it uses correct types for GPU usage.

Parameters:
    'g': the graph that will be normalized
"""
function normalize_graph(g::GNNGraph)
    s, t = edge_index(g)
    return GNNGraph(
        Int32.(s), Int32.(t); # Force correct typing!
        num_nodes = g.num_nodes,
        ndata     = g.ndata,
        edata     = g.edata,
        gdata     = g.gdata,
    )
end

"""
    Base.getindex(d::LazyGraphDataset, i::Int)

Returns a graph of a LazyGraphDataset with a given index.

Parameters:
    -'dataset': the dataset holding all graphs
    -'i': index of the graph
"""
function Base.getindex(dataset::LazyGraphDataset, i::Int)
    file = dataset.files[i]
    inst_path = joinpath(dataset.instances_dir, file)
    sol_path = joinpath(dataset.solutions_dir, replace(file, ".txt" => "_solution.txt"))
    
    raw_g = load_instance(inst_path)
    lines = readlines(sol_path)
    
    x_0 = zeros(Float32, 1, nv(raw_g))
    if length(lines) >= 3
        mask_str = split(lines[3])
        mask = parse.(Float32, mask_str)
        x_0 = reshape(mask, 1, :)
    end
    
    gnn_g = GNNGraph(raw_g, ndata=(; x=x_0))
    return (normalize_graph(gnn_g), file)
end

"""
    Base.getindex(d::LazyGraphDataset, i::Int)

Returns a batch of graphs from a LazyGraphDataset with a given vector of indices. 
Runs multithreaded to speed up batch-wise IO operations

Parameters:
    -'dataset': the dataset holding all graphs
    -'is': indeces of the graphs
"""
function Base.getindex(dataset::LazyGraphDataset, inds::AbstractVector{<:Integer})
    tasks = map(inds) do i
        Threads.@spawn dataset[i]
    end
    return fetch.(tasks)
end

"""
    load_lazy_collection(instance_collection::String)

Loads a collection of training data lazily.
Instead of loading the entire dataset into memory, this function provides a proxy of the dataset which will be fetched on time.

Parameters:
    -'instance_collection': name of the collection which will be fetched from the INSTANCES_PATH directory.
"""
function load_lazy_collection(instance_collection::String)
    collection_dir = joinpath(pkgdir(Thesis), ENV["INSTANCES_PATH"], instance_collection)
    instances_dir = joinpath(collection_dir, "instances")
    solutions_dir = joinpath(collection_dir, "solutions")
    inferences_dir = joinpath(collection_dir, "inferences")

    if !isdir(inferences_dir)
        mkpath(inferences_dir)
    end
    
    if !isdir(solutions_dir)
        @warn "Solutions directory not found. Are the instances solved yet?"
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

"""
    split_lazy_dataset(dataset::LazyGraphDataset, train_ratio::Float64=0.9)

Splits a LazyGraphDataset into a training and test set and returns both.

Parameters:
    -'dataset': the dataset which will be split
    -'train_ratio': the ratio of the training dataset. The rest will be testdata.
"""
function split_lazy_dataset(dataset::LazyGraphDataset, train_ratio::Float64=0.9)
    total = length(dataset)
    shuffled_files = shuffle(dataset.files)
    
    train_size = max(1, round(Int, train_ratio * total))
    
    train_files = shuffled_files[1:train_size]
    test_files = shuffled_files[train_size+1:end]
    
    train_dataset = LazyGraphDataset(dataset.instances_dir, dataset.solutions_dir, train_files)
    test_dataset = LazyGraphDataset(dataset.instances_dir, dataset.solutions_dir, test_files)
    
    return train_dataset, test_dataset
end

end