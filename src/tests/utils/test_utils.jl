# src/tests/utils/test_utils.jl
module TestUtils

using Flux
using GraphNeuralNetworks
using Random
using Printf
using Thesis
using ..InstanceUtils: load_instance

export load_collection, split_dataset, run_inference_and_save

"""
    load_collection(instance_collection::String)

Loads instances and solutions for the MISP dataset, returning a list of (GNNGraph, filename) tuples
and the target path for inferences.
"""
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

"""
    split_dataset(all_pairs, train_ratio=0.9)

Splits the dataset into training and testing datasets based on a train_ratio.
"""
function split_dataset(all_pairs::Vector{Tuple{GNNGraph, String}}, train_ratio::Float64=0.9)
    total_instances = length(all_pairs)
    test_size = max(1, round(Int, (1.0 - train_ratio) * total_instances))
    train_size = total_instances - test_size
    
    # Shuffle the pairs to ensure a random train/test split
    shuffled_pairs = shuffle(all_pairs)
    
    return shuffled_pairs[1:train_size], shuffled_pairs[train_size+1:end]
end

"""
    run_inference_and_save(model, test_pairs, inferences_dir)

Extracts all test graphs, passes them to `infer` at once, and handles 
the formatting and file saving for the resulting heatmaps.
"""
function run_inference_and_save(model::DiffusionModel, test_pairs, inferences_dir)
    # Extract just the graphs for the infer function
    graphs = [pair[1] for pair in test_pairs]
    filenames = [pair[2] for pair in test_pairs]
    
    inference_time = @elapsed begin
        # Call infer with all instances at once
        heatmaps_cpu = Thesis.infer(model, graphs)
    end
    
    @info "Inference for $(length(test_pairs)) graphs took $(round(inference_time, digits=4)) seconds."
    
    # Save the output for each instance
    for i in 1:length(test_pairs)
        g_cpu = graphs[i]
        filename = filenames[i]
        heatmap_cpu = heatmaps_cpu[i]
        
        num_nodes = g_cpu.num_nodes
        solution_binary = Int.(vec(g_cpu.ndata.x)) 
        heatmap_vec = vec(heatmap_cpu)
        
        sol_str = join(solution_binary, "  ")
        heatmap_str = join([@sprintf("%.1f", x) for x in heatmap_vec], " ")
        
        result_filename = replace(filename, ".txt" => "_result.txt")
        result_path = joinpath(inferences_dir, result_filename)
        
        open(result_path, "w") do io
            println(io, num_nodes)
            println(io, sol_str)
            println(io, heatmap_str)
        end
    end
end

end