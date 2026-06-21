# src/tests/utils/test_utils.jl

using Flux
using GraphNeuralNetworks
using Printf
using Dates
using Statistics
using Base.Iterators: partition

# Note: No 'module TestUtils' or 'export' needed anymore!

"""
    run_inference_and_save(model, test_pairs, inferences_dir)
"""
function run_inference_and_save(model::DiffusionModel, test_pairs, inferences_dir, batch_size=128)
    total_instances = length(test_pairs)

    @sync begin # for parallelism
        for i in 1:batch_size:total_instances
            end_idx = min(i + batch_size - 1, total_instances)

            batch = test_pairs[i:end_idx]
            graphs = [item[1] for item in batch]
            filenames = [item[2] for item in batch]
            node_counts = [g.num_nodes for g in graphs]

            # batch and infer
            batch_time = @elapsed begin
                heatmaps = Thesis.infer(model, Flux.batch(graphs))
            end

            actual_batch_size = length(batch)
            batch_num = div(i - 1, batch_size) + 1
            avg_time_per_instance = batch_time / actual_batch_size

            @info "Batch $batch_num [$(end_idx) instances total] | Total: $(round(batch_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

            # write in parallel
            Threads.@spawn begin
                start_slice = 1
                for j in 1:length(graphs)
                    num_nodes = node_counts[j]
                    end_slice = start_slice + num_nodes - 1

                    heatmap_slice = heatmaps[1, start_slice:end_slice]

                    solution_binary = Int.(vec(graphs[j].ndata.x))
                    heatmap_vec = vec(heatmap_slice)

                    sol_str = join(solution_binary, "  ")
                    heatmap_str = join([@sprintf("%.1f", x) for x in heatmap_vec], " ")

                    result_filename = replace(filenames[j], ".txt" => "_result.txt")
                    result_path = joinpath(inferences_dir, result_filename)

                    open(result_path, "w") do io
                        println(io, num_nodes)
                        println(io, sol_str)
                        println(io, heatmap_str)
                    end
                    start_slice = end_slice + 1
                end
            end
        end
    end
end

"""
    run_inference_greedy_and_save
"""
function run_inference_greedy_and_save(model::DiffusionModel, test_pairs, inferences_dir, batch_size=128)
    total_instances = length(test_pairs)

    @sync begin # for parallelism
        for i in 1:batch_size:total_instances
            end_idx = min(i + batch_size - 1, total_instances)

            batch = test_pairs[i:end_idx]
            graphs = [item[1] for item in batch] 
            filenames = [item[2] for item in batch]
            node_counts = [g.num_nodes for g in graphs]

            # batch and infer
            batch_time = @elapsed begin
                heatmaps = Thesis.infer(model, Flux.batch(graphs))
            end

            actual_batch_size = length(batch)
            batch_num = div(i - 1, batch_size) + 1
            avg_time_per_instance = batch_time / actual_batch_size

            @info "Batch $batch_num [$(end_idx) instances total] | Total: $(round(batch_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

            Threads.@spawn begin
                start_slice = 1
                for j in 1:length(graphs)
                    num_nodes = node_counts[j]
                    end_slice = start_slice + num_nodes - 1

                    heatmap_slice = heatmaps[1, start_slice:end_slice]
                    heatmap_vec = vec(heatmap_slice)

                    decoded_mask = Thesis.greedy_decode_mis(graphs[j], heatmap_vec)

                    solution_binary = Int.(vec(graphs[j].ndata.x))
                    decoded_binary = Int.(decoded_mask)
                    
                    sol_str = join(solution_binary, "  ")
                    decoded_str = join(decoded_binary, "  ")
                    heatmap_str = join([@sprintf("%.1f", x) for x in heatmap_vec], " ")

                    result_filename = replace(filenames[j], ".txt" => "_result.txt")
                    result_path = joinpath(inferences_dir, result_filename)

                    open(result_path, "w") do io
                        println(io, num_nodes)
                        println(io, sol_str)     
                        println(io, decoded_str) 
                        println(io, heatmap_str) 
                    end
                    
                    start_slice = end_slice + 1
                end
            end
        end
    end
end

"""
    run_automated_pipeline(...)
"""
function run_automated_pipeline(;
    run_id::String = Dates.format(Dates.now(), "yyyymmdd-HHMMSS"),
    generator::GraphGenerator,
    solver::ProblemSolver,
    gnn_config::GNNConfig,
    num_train::Int = 1000,
    num_test::Int = 100,
    epochs::Int = 5,
    batch_size::Int = 16,
    learning_rate::Float64 = 1e-3,
    diffusion_steps::Int = 100,
    inference_steps::Int = 20, 
    workers::Int = 12
)
    @info "=== Starting Pipeline Run: $run_id ==="

    @info "Initializing Train and Test datasets..."
    
    train_dataset = RuntimeDataset(num_train, generator, solver; max_concurrent_solves=workers)
    test_dataset = RuntimeDataset(num_test, generator, solver; max_concurrent_solves=workers)

    # Train uses collate=true for maximum GPU speed
    train_loader = Flux.DataLoader(train_dataset, batchsize=batch_size, shuffle=true, collate=true)
    
    # Test uses collate=false so we can easily access individual generated CPU graphs for saving!
    test_loader = Flux.DataLoader(test_dataset, batchsize=batch_size, shuffle=false, collate=false)

    gnn = DifuscoGNN(gnn_config)
    diffusion_process = BernoulliDiffusion(diffusion_steps, "cosine")
    model = DiffusionModel(gnn, diffusion_process) |> Flux.gpu

    num_params = sum(length, Flux.params(model.gnn))

    optim = Flux.setup(
        Flux.Optimisers.OptimiserChain(
            Flux.Optimisers.ClipGrad(1.0f0), 
            Flux.Adam(learning_rate)
        ),
        model.gnn
    )

    @info "Training model for $epochs epochs..."
    loss_history = Thesis.train_diffusion_model!(model, optim, train_loader, epochs=epochs)
    final_train_loss = isempty(loss_history) ? NaN : loss_history[end]

    @info "Testing model on $num_test unseen instances & streaming to disk..."
    total_model_score = 0.0
    total_true_score = 0.0
    total_graphs = 0

    # Create directory for the streaming test instances
    test_results_dir = joinpath(pkgdir(Thesis), "results", "test_instances_$run_id")
    mkpath(test_results_dir)

    Flux.testmode!(model.gnn)

    for batch in test_loader
        # 1. Extract the raw CPU graphs generated on-the-fly
        graphs = [item[1] for item in batch]
        names = [item[2] for item in batch]
        
        # 2. Batch them for the GPU
        g_batch = Flux.batch(graphs)
        g_batch_gpu = g_batch |> Flux.gpu
        target_x0 = g_batch_gpu.ndata.x 

        # 3. Predict Heatmap
        shape = size(target_x0)
        preds_probs = Thesis.sample_heatmap(model.method, model, shape, g_batch_gpu; inference_steps=inference_steps)
        
        true_mask = target_x0 .> 0.5f0

        # 4. Bring Heatmap back to CPU for Greedy Decoding
        preds_probs_cpu = Flux.cpu(preds_probs)
        
        # --- NEW: Synchronous Greedy Decoding for Metrics ---
        valid_masks_cpu = Vector{Vector{Int}}()
        batch_model_score = 0.0
        
        start_slice = 1
        for j in 1:length(graphs)
            num_nodes = graphs[j].num_nodes
            end_slice = start_slice + num_nodes - 1
            
            heatmap_slice = preds_probs_cpu[1, start_slice:end_slice]
            
            # Apply Greedy Decoder to ensure validity
            mask_slice = Thesis.greedy_decode_mis(graphs[j], vec(heatmap_slice))
            push!(valid_masks_cpu, mask_slice)
            
            # Sum the valid greedy size!
            batch_model_score += sum(mask_slice)
            
            start_slice = end_slice + 1
        end

        # Accumulate the CORRECT valid scores
        total_model_score += batch_model_score
        total_true_score += Float64(sum(true_mask))
        total_graphs += length(graphs)

        # 5. Stream to disk asynchronously 
        # (Pass the already-decoded valid_masks_cpu to the background thread!)
        let local_graphs = graphs, local_names = names, local_probs = preds_probs_cpu, local_masks = valid_masks_cpu
            Threads.@spawn begin
                start_idx = 1
                for j in 1:length(local_graphs)
                    g_single = local_graphs[j]
                    num_nodes = g_single.num_nodes
                    end_idx = start_idx + num_nodes - 1
                    
                    heatmap_slice = local_probs[1, start_idx:end_idx]
                    mask_slice = local_masks[j] # Uses the valid greedy mask calculated above!
                    
                    # Extract graph edges
                    s, t = edge_index(g_single)
                    num_edges = div(length(s), 2)
                    
                    filepath = joinpath(test_results_dir, replace(local_names[j], ".txt" => "_result.txt"))
                    
                    open(filepath, "w") do io
                        # Topography Block
                        println(io, "$num_nodes $num_edges")
                        for k in 1:length(s)
                            if s[k] < t[k] # Only write undirected edges once
                                println(io, "$(s[k]) $(t[k])")
                            end
                        end
                        
                        # Data Block
                        println(io, join(Int.(vec(g_single.ndata.x)), " "))   # Ground Truth
                        println(io, join(Int.(vec(mask_slice)), " "))         # Greedy Predicted Mask
                        println(io, join([@sprintf("%.4f", x) for x in vec(heatmap_slice)], " ")) # Heatmap
                    end
                    
                    start_idx = end_idx + 1
                end
            end
        end
    end

    Flux.trainmode!(model.gnn)

    avg_model_score = total_model_score / total_graphs
    avg_true_score = total_true_score / total_graphs
    performance_pct = (avg_model_score / avg_true_score) * 100

    @info "Testing Complete. Avg Model Score: $(round(avg_model_score, digits=2)) | Avg Optimal Score: $(round(avg_true_score, digits=2)) | Performance: $(round(performance_pct, digits=2))%"

    # 6. Write Meta-Results to File
    results_dir = joinpath(pkgdir(Thesis), "results")
    mkpath(results_dir)
    filepath = joinpath(results_dir, "report-$run_id.txt")

    open(filepath, "w") do io
        println(io, "==================================================")
        println(io, "   PIPELINE RUN: $run_id              ")
        println(io, "==================================================")
        println(io, "\n[ DATA CONFIGURATION ]")
        println(io, "Graph Generator: ", typeof(generator))
        println(io, "Generator Params: ", generator) 
        println(io, "Problem Solver:  ", typeof(solver))
        println(io, "Solver Params:   ", solver)
        println(io, "Training Instances: ", num_train)
        println(io, "Testing Instances:  ", num_test)
        
        println(io, "\n[ MODEL CONFIGURATION ]")
        println(io, "GNN Layers: ", gnn_config.num_layers)
        println(io, "GNN Hidden Dim: ", gnn_config.hidden_dim)
        println(io, "Total Parameters: ", num_params)
        println(io, "Diffusion Steps: ", diffusion_steps)
        println(io, "Inference Steps: ", inference_steps)
        
        println(io, "\n[ TRAINING HYPERPARAMETERS ]")
        println(io, "Epochs: ", epochs)
        println(io, "Batch Size: ", batch_size)
        println(io, "Learning Rate: ", learning_rate)
        
        println(io, "\n[ RESULTS & METRICS ]")
        println(io, "Final Training Loss: ", round(final_train_loss, digits=6))
        println(io, "--------------------------------------------------")
        println(io, "Avg Optimal MIS Size:   ", round(avg_true_score, digits=2))
        println(io, "Avg Predicted MIS Size: ", round(avg_model_score, digits=2))
        println(io, "Approximation Ratio:    ", round(performance_pct, digits=2), "%")
        println(io, "==================================================")
    end

    @info "Report saved to $filepath"
    
    # 7. Save the Trained Model
    @info "Saving trained model weights..."
    models_dir = joinpath(pkgdir(Thesis), "models") 
    mkpath(models_dir)
    model_path = joinpath(models_dir, "model-$run_id.jld2")
    
    Thesis.save_model(model, model_path)
    @info "Model successfully saved to $model_path"

    return filepath
end