# src/tests/utils/test_utils.jl
module TestUtils

using Flux
using GraphNeuralNetworks
using Random
using Printf
using Thesis
using ..InstanceUtils: load_instance

export load_collection, split_dataset, train_diffusion_model!, run_inference_and_save

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
    train_diffusion_model!(model, optim, training_pairs; epochs=5)

Standard training loop for diffusion models, optimizing the MSE loss for predicted noise.
"""
function train_diffusion_model!(model::DiffusionModel, optim, training_pairs; epochs::Int=5)
    loss_history = Float32[]
    for epoch in 1:epochs
        epoch_loss = 0.0f0
        shuffled_data = training_pairs[randperm(length(training_pairs))]
        
        for (g, _) in shuffled_data
            x_start = g.ndata.x 
            val, grads = Flux.withgradient(model) do m
                Thesis.compute_loss(m.process, m, x_start, g)
            end
            Flux.update!(optim, model, grads[1])
            epoch_loss += val
        end
        
        avg_loss = epoch_loss / length(training_pairs)
        push!(loss_history, avg_loss)
        @info "Epoch $epoch | Avg Loss: $avg_loss"
    end
    return loss_history
end

"""
    run_inference_and_save(model, test_pairs, inferences_dir)

Generates heatmaps for the test set and writes the outputs to the inferences directory.
"""
function run_inference_and_save(model::DiffusionModel, test_pairs, inferences_dir)
    for (g, filename) in test_pairs
        num_nodes = g.num_nodes
        shape = (1, num_nodes)
        
        heatmap = Thesis.sample_heatmap(model.process, model, shape, g)
        
        solution_binary = Int.(vec(g.ndata.x)) 
        heatmap_vec = vec(heatmap)
        
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