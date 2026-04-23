# src/tests/utils/test_utils.jl
module TestUtils

using Flux
using GraphNeuralNetworks
using Printf
using Thesis
using Printf
using Base.Iterators: partition

export run_inference_and_save, run_inference_greedy_and_save

"""
    run_inference_and_save(model, test_pairs, inferences_dir)

Extracts all test graphs, passes them to infer and saves the resulting heatmaps.
Runs multithreaded to speed-up IO operations!

Parameters
    -'model': the trained model
    -'test_pairs': testdata
    -'inferences_dir': directory to save results to
    -'batch_size': batch size for inferences
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

            if i == 3
                #@profview Thesis.infer(model, Flux.batch(graphs))
            end

            # batch and infer
            batch_time = @elapsed begin
                heatmaps = Thesis.infer(model, Flux.batch(graphs))
            end

            avg_time_per_instance = batch_time / batch_size

            # for logging only
            actual_batch_size = length(batch)
            batch_num = div(i - 1, batch_size) + 1
            avg_time_per_instance = batch_time / actual_batch_size

            @info "Batch $batch_num [$(end_idx) instances total] | Total: $(round(batch_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

            # write in parallel (for each graph within a batch) to optimize IO speed
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

    @sync begin # for parallelism
        for i in 1:batch_size:total_instances
            end_idx = min(i + batch_size - 1, total_instances)

            batch = test_pairs[i:end_idx]
            graphs = [item[1] for item in batch] # These are still on the CPU!
            filenames = [item[2] for item in batch]
            node_counts = [g.num_nodes for g in graphs]

            # batch and infer (Thesis.infer handles moving the batch to GPU and returning the heatmap to CPU)
            batch_time = @elapsed begin
                heatmaps = Thesis.infer(model, Flux.batch(graphs))
            end

            actual_batch_size = length(batch)
            batch_num = div(i - 1, batch_size) + 1
            avg_time_per_instance = batch_time / actual_batch_size

            @info "Batch $batch_num [$(end_idx) instances total] | Total: $(round(batch_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

            # write in parallel to optimize IO speed
            Threads.@spawn begin
                start_slice = 1
                for j in 1:length(graphs)
                    num_nodes = node_counts[j]
                    end_slice = start_slice + num_nodes - 1

                    # Extract the heatmap slice for this specific graph
                    heatmap_slice = heatmaps[1, start_slice:end_slice]
                    heatmap_vec = vec(heatmap_slice)

                    # --- NEW: Apply Greedy Decoding ---
                    # Both graphs[j] and heatmap_vec are on the CPU, perfectly matching the function signature
                    decoded_mask = Thesis.greedy_decode_mis(graphs[j], heatmap_vec)

                    # Format strings for saving
                    solution_binary = Int.(vec(graphs[j].ndata.x))
                    decoded_binary = Int.(decoded_mask)
                    
                    sol_str = join(solution_binary, "  ")
                    decoded_str = join(decoded_binary, "  ")
                    heatmap_str = join([@sprintf("%.1f", x) for x in heatmap_vec], " ")

                    result_filename = replace(filenames[j], ".txt" => "_result.txt")
                    result_path = joinpath(inferences_dir, result_filename)

                    open(result_path, "w") do io
                        println(io, num_nodes)
                        println(io, sol_str)     # Ground truth
                        println(io, decoded_str) # Greedy decoded result
                        println(io, heatmap_str) # Heatmap probabilities
                    end
                    
                    start_slice = end_slice + 1
                end
            end
        end
    end
end

end