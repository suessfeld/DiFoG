using Flux
using GraphNeuralNetworks
using Printf
using Dates
using Statistics
using Base.Iterators: partition

function run_inference_and_save(model::DiffusionModel, test_pairs, inferences_dir, batch_size=128)
    total_instances = length(test_pairs)

    @sync begin
        for i in 1:batch_size:total_instances
            end_idx = min(i + batch_size - 1, total_instances)

            batch = test_pairs[i:end_idx]
            graphs = [item[1] for item in batch]
            filenames = [item[2] for item in batch]
            node_counts = [g.num_nodes for g in graphs]

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

function run_inference_greedy_and_save(model::DiffusionModel, test_pairs, inferences_dir, batch_size=128)
    total_instances = length(test_pairs)

    @sync begin
        for i in 1:batch_size:total_instances
            end_idx = min(i + batch_size - 1, total_instances)

            batch = test_pairs[i:end_idx]
            graphs = [item[1] for item in batch] 
            filenames = [item[2] for item in batch]
            node_counts = [g.num_nodes for g in graphs]

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

function run_automated_pipeline(;
    run_id::String = Dates.format(Dates.now(), "yyyymmdd-HHMMSS"),
    model::DiffusionModel,
    generator::GraphGenerator,
    solver::ProblemSolver,
    num_train::Int = 1000,
    num_test::Int = 100,
    epochs::Int = 5,
    batch_size::Int = 16,
    learning_rate::Float64 = 1e-3,
    workers::Int = 12,
    test_time_limit::Float64 = 2.0,
)
    @info "=== Starting Pipeline Run: $run_id ==="

    models_dir = joinpath(pkgdir(Thesis), "models")
    mkpath(models_dir)
    model_path = joinpath(models_dir, "model-$run_id.jld2")

    model_already_exists = isfile(model_path)

    test_solver = solver isa MISPSolver ? MISPSolver(time_limit=test_time_limit) : solver

    final_train_loss = NaN

    if model_already_exists
        @info "Found existing model for run_id '$run_id' at $model_path — loading weights and skipping training."
        Thesis.load_model!(model, model_path)
    else
        @info "No existing model found for run_id '$run_id' — training from scratch."

        @info "Initializing Train dataset..."
        train_dataset = RuntimeDataset(num_train, generator, solver; max_concurrent_solves=workers)
        train_loader = Flux.DataLoader(train_dataset, batchsize=batch_size, shuffle=true, collate=true)

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
    end

    num_params = sum(length, Flux.params(model.gnn))

    @info "Initializing Test dataset (Gurobi time limit: $(test_time_limit)s)..."
    test_dataset = RuntimeDataset(num_test, generator, test_solver; max_concurrent_solves=workers)
    test_loader = Flux.DataLoader(test_dataset, batchsize=batch_size, shuffle=false, collate=false)

     @info "Testing model on $num_test unseen instances & streaming to disk..."
    total_model_score = 0.0
    total_true_score = 0.0
    total_graphs = 0
    total_inference_time = 0.0

    test_results_dir = joinpath(pkgdir(Thesis), "results", "test_instances_$run_id")
    mkpath(test_results_dir)

    Flux.testmode!(model.gnn)

    @sync begin
        global_test_idx = 1
        batch_num = 0

        for batch in test_loader
            graphs = [item[1] for item in batch]
            names = ["test_instance_$(global_test_idx + k - 1).txt" for k in 1:length(graphs)]
            global_test_idx += length(graphs)

            g_batch = Flux.batch(graphs)
            g_batch_gpu = g_batch |> Flux.gpu
            target_x0 = g_batch_gpu.ndata.x

            shape = size(target_x0)
            batch_size_actual = length(graphs)

            batch_inference_time = @elapsed begin
                preds_probs = Thesis.sample_heatmap(model.method, model, shape, g_batch_gpu)
                CUDA.synchronize()
            end
            total_inference_time += batch_inference_time
            avg_time_per_instance = batch_inference_time / batch_size_actual

            batch_num += 1
            @info "Batch $(batch_num) [$(total_graphs + batch_size_actual) instances total] | Total: $(round(batch_inference_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

            true_mask = target_x0 .> 0.5f0
            preds_probs_cpu = Flux.cpu(preds_probs)

            valid_masks_cpu = Vector{Vector{Int}}()
            batch_model_score = 0.0

            start_slice = 1
            for j in 1:length(graphs)
                num_nodes = graphs[j].num_nodes
                end_slice = start_slice + num_nodes - 1

                heatmap_slice = preds_probs_cpu[1, start_slice:end_slice]
                mask_slice = Thesis.greedy_decode_mis(graphs[j], vec(heatmap_slice))
                push!(valid_masks_cpu, mask_slice)

                batch_model_score += sum(mask_slice)
                start_slice = end_slice + 1
            end

            total_model_score += batch_model_score
            total_true_score += Float64(sum(true_mask))
            total_graphs += batch_size_actual

            let local_graphs = graphs, local_names = names, local_probs = preds_probs_cpu, local_masks = valid_masks_cpu
                Threads.@spawn begin
                    start_idx = 1
                    for j in 1:length(local_graphs)
                        g_single = local_graphs[j]
                        num_nodes = g_single.num_nodes
                        end_idx = start_idx + num_nodes - 1

                        heatmap_slice = local_probs[1, start_idx:end_idx]
                        mask_slice = local_masks[j]

                        s, t = edge_index(g_single)
                        num_edges = div(length(s), 2)

                        filepath = joinpath(test_results_dir, replace(local_names[j], ".txt" => "_result.txt"))

                        open(filepath, "w") do io
                            println(io, "$num_nodes $num_edges")
                            for k in 1:length(s)
                                if s[k] < t[k]
                                    println(io, "$(s[k]) $(t[k])")
                                end
                            end
                            println(io, join(Int.(vec(g_single.ndata.x)), " "))
                            println(io, join(Int.(vec(mask_slice)), " "))
                            println(io, join([@sprintf("%.4f", x) for x in vec(heatmap_slice)], " "))
                        end

                        start_idx = end_idx + 1
                    end
                end
            end
        end
    end

    Flux.trainmode!(model.gnn)

    avg_model_score = total_model_score / total_graphs
    avg_true_score = total_true_score / total_graphs
    performance_pct = (avg_model_score / avg_true_score) * 100
    avg_inference_time_per_instance = total_inference_time / total_graphs

    @info "Testing Complete. Avg Model Score: $(round(avg_model_score, digits=2)) | Avg Optimal Score: $(round(avg_true_score, digits=2)) | Performance: $(round(performance_pct, digits=2))% | Avg Inference Time: $(round(avg_inference_time_per_instance, digits=4))s/instance"

    results_dir = joinpath(pkgdir(Thesis), "results")
    mkpath(results_dir)
    filepath = joinpath(results_dir, "report-$run_id.txt")

    if !model_already_exists
        @info "Saving trained model weights..."
        Thesis.save_model(model, model_path)
        @info "Model successfully saved to $model_path"
    else
        @info "Skipped saving — model was loaded from disk, weights unchanged."
    end

    open(filepath, "w") do io
        println(io, "==================================================")
        println(io, "   PIPELINE RUN: $run_id              ")
        println(io, "==================================================")
        println(io, "\n[ DATA CONFIGURATION ]")
        println(io, "Graph Generator: ", typeof(generator))
        println(io, "Generator Params: ", generator)
        println(io, "Problem Solver (train):  ", typeof(solver), " ", solver)
        println(io, "Problem Solver (test):   ", typeof(test_solver), " ", test_solver)
        println(io, "Training Instances: ", num_train)
        println(io, "Testing Instances:  ", num_test)

        println(io, "\n[ MODEL CONFIGURATION ]")
        println(io, "Method: ", typeof(model.method))
        println(io, "GNN Layers: ", model.gnn.config.num_layers)
        println(io, "GNN Hidden Dim: ", model.gnn.config.hidden_dim)
        println(io, "Total Parameters: ", num_params)
        println(io, "Timesteps (T): ", model.method.scheduler.total_steps)
        println(io, "Inference Steps: ", model.method.scheduler.inference_steps)

        println(io, "\n[ TRAINING HYPERPARAMETERS ]")
        println(io, "Loaded pretrained model: ", model_already_exists)
        println(io, "Epochs: ", epochs)
        println(io, "Batch Size: ", batch_size)
        println(io, "Learning Rate: ", learning_rate)

        println(io, "\n[ RESULTS & METRICS ]")
        println(io, "Final Training Loss: ", model_already_exists ? "N/A (loaded)" : round(final_train_loss, digits=6))
        println(io, "--------------------------------------------------")
        println(io, "Avg Optimal MIS Size:   ", round(avg_true_score, digits=2))
        println(io, "Avg Predicted MIS Size: ", round(avg_model_score, digits=2))
        println(io, "Approximation Ratio:    ", round(performance_pct, digits=2), "%")
        println(io, "--------------------------------------------------")
        println(io, "Total Inference Time (sampling only): ", round(total_inference_time, digits=4), "s")
        println(io, "Avg Inference Time per Instance:       ", round(avg_inference_time_per_instance, digits=4), "s")
        println(io, "==================================================")
    end

    @info "Report saved to $filepath"

    if !model_already_exists
        @info "Saving trained model weights..."
        Thesis.save_model(model, model_path)
        @info "Model successfully saved to $model_path"
    else
        @info "Skipped saving — model was loaded from disk, weights unchanged."
    end

    return filepath
end