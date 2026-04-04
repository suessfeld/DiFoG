# src/tests/utils/test_utils.jl
module TestUtils

using Flux
using GraphNeuralNetworks
using Printf
using Thesis
using Printf
using Base.Iterators: partition

export run_inference_and_save

"""
    run_inference_and_save(model, test_pairs, inferences_dir)

Extracts all test graphs, passes them to `infer` at once (or one-by-one lazily), 
and handles the formatting and file saving for the resulting heatmaps.
"""
function run_inference_and_save(model::DiffusionModel, test_pairs, inferences_dir; batch_size=64)
    inference_time = @elapsed begin
        total_instances = length(test_pairs)
        
        # @sync guarantees that Julia waits for all background file-writing 
        # threads to finish before exiting the function!
        @sync begin
            for i in 1:batch_size:total_instances
                end_idx = min(i + batch_size - 1, total_instances)
                
                # This explicitly triggers your 16-thread getindex function!
                batch = test_pairs[i:end_idx]
                
                graphs = [item[1] for item in batch]
                filenames = [item[2] for item in batch]
                node_counts = [g.num_nodes for g in graphs]
                
                # 1. Glue the graphs together into one giant disjoint graph
                g_batch = Flux.batch(graphs)
                
                # 2. INFERENCE: A single, highly optimized GPU call for the entire batch
                heatmap_batch_cpu = Thesis.infer(model, g_batch)
                
                # 3. PARALLEL I/O: Spawn a background task to format and write to disk
                Threads.@spawn begin
                    start_slice = 1
                    for j in 1:length(graphs)
                        num_nodes = node_counts[j]
                        end_slice = start_slice + num_nodes - 1
                        
                        # Extract the exact slice of the heatmap belonging to this specific graph
                        heatmap_slice = heatmap_batch_cpu[1, start_slice:end_slice]
                        
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
                        
                        # Shift the index for the next graph in the batch
                        start_slice = end_slice + 1
                    end
                end
            end
        end
    end
    
    @info "Inference for $(total_instances) graphs took $(round(inference_time, digits=4)) seconds."
end

end