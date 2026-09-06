using Flux
using GraphNeuralNetworks
using Printf
using Dates
using Statistics
using Base.Iterators: partition

"""
    run_automated_pipeline(;run_id::String = Dates.format(Dates.now(), "yyyymmdd-HHMMSS"),model::DiffusionModel,generator::GraphGenerator,solver::ProblemSolver,num_train::Int = 1000,num_test::Int = 100,num_val::Int = max(20, div(num_train,20)),epochs::Int = 5,batch_size::Int = 16,learning_rate::Float64 = 1e-3,workers::Int = 12,test_time_limit::Float64 = 2.0,num_repeat_inference::Int = 5,ood_generator=nothing,ood_num_test::Int=50

A helper function that runs a training and test run, then evaluates the models and saves the results.
    
Parameters:
    - 'run_id': The id of this run. Defaults to this timestamp. Before training it is checked whether a model with the provided run_id is already saved, which will be loaded.
    - 'model': The model that should be trained and tested.
    - 'generator': The used graph generator.
    - 'solver': The solver that solves the problem.
    - 'num_train': The number of training instances.
    - 'num_test': The number of testing instances.
    - 'num_val': The number of held-out validation instances, materialized once (fixed across epochs) before training starts and used to report per-epoch validation loss. Set to 0 to disable validation entirely (B7).
    - 'epochs': The amount of epochs that will be trained for.
    - 'batch_size': The batch size that will be used during training and testing.
    - 'learning_rate': The learning rate used for the training
    - 'workers': The number of workes used for solving the instances.
    - 'test_time_limit': The timelimit for the solver during the testing stage.
    - 'num_repeat_inference': The amount of inferences that will be done on the last instance. This is to analye the behavior of inferencing on the same instance multiple times.
    - 'ood_generator': Optional second `GraphGenerator` (e.g. configured for larger/different graphs) used to run a supplementary out-of-distribution evaluation pass after the main in-distribution test (B7). `nothing` (default) skips this entirely.
    - 'ood_num_test': Number of instances drawn from `ood_generator` for the supplementary pass, if given.
"""

function run_automated_pipeline(;
    run_id::String = Dates.format(Dates.now(), "yyyymmdd-HHMMSS"),
    model::DiffusionModel,
    generator::GraphGenerator,
    solver::ProblemSolver,
    num_train::Int = 1000,
    num_test::Int = 100,
    num_val::Int = max(20, div(num_train, 20)),
    epochs::Int = 5,
    batch_size::Int = 16,
    learning_rate::Float64 = 1e-3,
    workers::Int = 12,
    test_time_limit::Float64 = 5.0,
    num_repeat_inference::Int = 5,
    ood_generator::Union{Nothing, GraphGenerator} = nothing,
    ood_num_test::Int = 50,
)
    @info "=== Starting Pipeline Run: $run_id ==="

    models_dir = joinpath(pkgdir(Thesis), "models")
    mkpath(models_dir)
    model_path = joinpath(models_dir, "model-$run_id.jld2")

    model_already_exists = isfile(model_path)

    test_solver = if solver isa MISPSolver
        MISPSolver(time_limit=test_time_limit)
    elseif solver isa MaxCutSolver
        MaxCutSolver(time_limit=test_time_limit)
    elseif solver isa MaxDegSolver
        solver
    else
        error("run_automated_pipeline: no test_solver mapping defined for $(typeof(solver))")
    end

    final_train_loss = NaN
    final_val_loss = NaN
    val_loader = nothing

    if model_already_exists
        @info "Found existing model for run_id '$run_id' at $model_path — loading weights and skipping training."
        Thesis.load_model!(model, model_path)
    else
        @info "No existing model found for run_id '$run_id' — training from scratch."

        if num_val > 0

            @info "Building fixed validation set ($(num_val) instances, held constant across epochs)..."
            val_instances = Vector{Tuple{GNNGraph,String}}(undef, num_val)
            for i in 1:num_val
                g_val = generator()
                x0_val = solver(g_val)
                gnn_g_val = GNNGraph(g_val, ndata=(; x=x0_val))
                val_instances[i] = (Thesis.normalize_graph(gnn_g_val), "val_instance_$(i).txt")
            end
            val_loader = Flux.DataLoader(val_instances, batchsize=batch_size, shuffle=false, collate=true)
        else
            @info "num_val=0 — validation loss tracking disabled for this run."
        end

        @info "Initializing Train dataset..."
        train_dataset = RuntimeDataset(num_train, generator, solver; max_concurrent_solves=workers)
        train_loader = Flux.DataLoader(train_dataset, batchsize=batch_size, shuffle=false, collate=true)

        optim = Flux.setup(
            Flux.Optimisers.OptimiserChain(
                Flux.Optimisers.ClipGrad(1.0f0),
                Flux.Adam(learning_rate)
            ),
            model.gnn
        )

        @info "Training model for $epochs epochs..."
        loss_history, val_loss_history = Thesis.train_diffusion_model!(model, optim, train_loader; epochs=epochs, val_loader=val_loader)
        final_train_loss = isempty(loss_history) ? NaN : loss_history[end]
        final_val_loss = isempty(val_loss_history) ? NaN : val_loss_history[end]

        @info "Saving trained model weights immediately after training completes (before test/eval, so a later crash can't discard a completed training run)..."
        Thesis.save_model(model, model_path)
        @info "Model successfully saved to $model_path"
    end

    num_params = sum(length, Flux.trainables(model.gnn))

    @info "Initializing Test dataset (Gurobi time limit: $(test_time_limit)s)..."
    test_dataset = RuntimeDataset(num_test, generator, test_solver; max_concurrent_solves=workers)
    test_loader = Flux.DataLoader(test_dataset, batchsize=batch_size, shuffle=false, collate=false)

    @info "Testing model on $num_test unseen instances & streaming to disk..."
    total_model_score = 0.0
    total_true_score = 0.0
    total_graphs = 0
    total_inference_time = 0.0

    total_confidence = 0.0
    total_entropy = 0.0
    total_correction_rate = 0.0

    per_instance_model_scores = Float64[]
    per_instance_true_scores = Float64[]

    last_test_graph = nothing
    last_test_instance_idx = 0

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

            true_mask_cpu = Flux.cpu(target_x0) .> 0.5f0
            preds_probs_cpu = Flux.cpu(preds_probs)

            valid_masks_cpu = Vector{Vector{Int}}()
            batch_model_score = 0.0
            batch_true_score = 0.0

            start_slice = 1
            for j in 1:length(graphs)
                num_nodes = graphs[j].num_nodes
                end_slice = start_slice + num_nodes - 1

                heatmap_slice = preds_probs_cpu[1, start_slice:end_slice]
                heatmap_vec = vec(heatmap_slice)
                mask_slice = Thesis.decode_heatmap(test_solver, graphs[j], heatmap_vec)
                push!(valid_masks_cpu, mask_slice)

                model_score_j = Thesis.evaluate(test_solver, graphs[j], mask_slice)
                true_score_j  = Thesis.evaluate(test_solver, graphs[j], vec(true_mask_cpu[1, start_slice:end_slice]))
                batch_model_score += model_score_j
                batch_true_score  += true_score_j
                push!(per_instance_model_scores, model_score_j)
                push!(per_instance_true_scores, true_score_j)

                # NEW — heatmap decisiveness (mean binary entropy → confidence score)
                mean_entropy, confidence = Thesis.heatmap_decisiveness(heatmap_vec)
                total_entropy += mean_entropy
                total_confidence += confidence

                # NEW — correction distance (naive threshold vs feasible greedy mask), normalized by node count
                corr_dist = Thesis.correction_distance(heatmap_vec, mask_slice)
                total_correction_rate += corr_dist / num_nodes

                start_slice = end_slice + 1
            end

            total_model_score += batch_model_score
            total_true_score += batch_true_score
            total_graphs += batch_size_actual


            last_test_graph = graphs[end]
            last_test_instance_idx = global_test_idx - 1

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

    # generic evaluation parameters
    avg_model_score = total_model_score / total_graphs
    avg_true_score = total_true_score / total_graphs
    performance_pct = (avg_model_score / avg_true_score) * 100
    avg_inference_time_per_instance = total_inference_time / total_graphs

    avg_confidence = total_confidence / total_graphs
    avg_entropy = total_entropy / total_graphs
    avg_correction_rate = total_correction_rate / total_graphs

    score_ratios = Float64[]
    for (m, t) in zip(per_instance_model_scores, per_instance_true_scores)
        if t > 0
            push!(score_ratios, m / t * 100)
        end
    end
    approx_ratio_std = length(score_ratios) > 1 ? std(score_ratios) : NaN
    approx_ratio_ci95 = length(score_ratios) > 1 ? 1.96 * approx_ratio_std / sqrt(length(score_ratios)) : NaN

    @info "Testing Complete. Avg Model Score: $(round(avg_model_score, digits=2)) | Avg Optimal Score: $(round(avg_true_score, digits=2)) | Performance: $(round(performance_pct, digits=2))% ± $(round(approx_ratio_ci95, digits=2))% (95% CI, std=$(round(approx_ratio_std, digits=2))%) | Avg Inference Time: $(round(avg_inference_time_per_instance, digits=4))s/instance"
    @info "Heatmap Quality | Avg Decisiveness: $(round(avg_confidence, digits=4)) (H̄=$(round(avg_entropy, digits=4))) | Avg Greedy Correction Rate: $(round(avg_correction_rate, digits=4))"

    avg_hamming_stability = NaN
    if num_repeat_inference > 1 && !isnothing(last_test_graph)
        @info "Running $(num_repeat_inference) repeated inferences on instance $(last_test_instance_idx) for stability measurement..."

        g_repeat_gpu = last_test_graph |> Flux.gpu
        shape = (1, g_repeat_gpu.num_nodes)

        s, t = edge_index(last_test_graph)
        num_edges = div(length(s), 2)
        gt_solution = Int.(vec(last_test_graph.ndata.x))

        repeat_masks = Vector{Vector{Int}}(undef, num_repeat_inference)

        for r in 1:num_repeat_inference
            heatmap_r = Thesis.sample_heatmap(model.method, model, shape, g_repeat_gpu)
            heatmap_r_cpu = vec(Flux.cpu(heatmap_r))
            mask_r = Int.(vec(Thesis.decode_heatmap(test_solver, last_test_graph, heatmap_r_cpu)))
            repeat_masks[r] = mask_r

            filepath = joinpath(test_results_dir, "test_instance_$(last_test_instance_idx)_$(r)_result.txt")
            open(filepath, "w") do io
                println(io, "$(last_test_graph.num_nodes) $num_edges")
                for k in 1:length(s)
                    if s[k] < t[k]
                        println(io, "$(s[k]) $(t[k])")
                    end
                end
                println(io, join(gt_solution, " "))
                println(io, join(mask_r, " "))
                println(io, join([@sprintf("%.4f", x) for x in heatmap_r_cpu], " "))
            end
        end

        pairwise_distances = Float64[]
        for i in 1:num_repeat_inference, j in (i+1):num_repeat_inference
            hd = Thesis.hamming_distance(repeat_masks[i], repeat_masks[j])
            push!(pairwise_distances, hd / last_test_graph.num_nodes)
        end
        avg_hamming_stability = isempty(pairwise_distances) ? 0.0 : mean(pairwise_distances)

        @info "Repeated Inference Stability | Instance $(last_test_instance_idx) | Avg Normalized Hamming Distance across $(num_repeat_inference) runs: $(round(avg_hamming_stability, digits=4))"
    end

    ood_avg_model_score = NaN
    ood_avg_true_score = NaN
    ood_performance_pct = NaN
    if !isnothing(ood_generator)
        @info "Running supplementary out-of-distribution evaluation on $(ood_num_test) instances from $(typeof(ood_generator))..."
        ood_test_dataset = RuntimeDataset(ood_num_test, ood_generator, test_solver; max_concurrent_solves=workers)
        ood_test_loader = Flux.DataLoader(ood_test_dataset, batchsize=batch_size, shuffle=false, collate=false)

        ood_total_model_score = 0.0
        ood_total_true_score = 0.0
        ood_total_graphs = 0

        for batch in ood_test_loader
            graphs = [item[1] for item in batch]
            g_batch = Flux.batch(graphs)
            g_batch_gpu = g_batch |> Flux.gpu
            target_x0 = g_batch_gpu.ndata.x
            shape = size(target_x0)

            preds_probs = Thesis.sample_heatmap(model.method, model, shape, g_batch_gpu)
            preds_probs_cpu = Flux.cpu(preds_probs)
            true_mask_cpu = Flux.cpu(target_x0) .> 0.5f0

            start_slice = 1
            for j in 1:length(graphs)
                num_nodes = graphs[j].num_nodes
                end_slice = start_slice + num_nodes - 1
                heatmap_vec = vec(preds_probs_cpu[1, start_slice:end_slice])
                mask_slice = Thesis.decode_heatmap(test_solver, graphs[j], heatmap_vec)

                ood_total_model_score += Thesis.evaluate(test_solver, graphs[j], mask_slice)
                ood_total_true_score  += Thesis.evaluate(test_solver, graphs[j], vec(true_mask_cpu[1, start_slice:end_slice]))
                ood_total_graphs += 1

                start_slice = end_slice + 1
            end
        end

        ood_avg_model_score = ood_total_model_score / ood_total_graphs
        ood_avg_true_score = ood_total_true_score / ood_total_graphs
        ood_performance_pct = (ood_avg_model_score / ood_avg_true_score) * 100
        @info "OOD Evaluation Complete. Avg Model Score: $(round(ood_avg_model_score, digits=2)) | Avg Optimal Score: $(round(ood_avg_true_score, digits=2)) | Performance: $(round(ood_performance_pct, digits=2))%"
    end

    Flux.trainmode!(model.gnn)

    results_dir = joinpath(pkgdir(Thesis), "results")
    mkpath(results_dir)
    filepath = joinpath(results_dir, "report-$run_id.tex")


    label_base = replace(run_id, r"[^A-Za-z0-9]" => "-")


    # output directly in latex tables
    open(filepath, "w") do io
        println(io, "% Auto-generated pipeline report — copy directly into thesis .tex source")
        println(io, "\\subsection{Pipeline Run: $(latex_escape(run_id))}")
        println(io, "\\label{sec:run-$(label_base)}")
        println(io)

        # ---- Data Configuration ----
        println(io, "\\subsubsection{Data Configuration}")
        write_latex_kv_table(io, [
            ("Graph Generator", string(typeof(generator))),
            ("Generator Params", string(generator)),
            ("Problem Solver (train)", "$(typeof(solver)) $(solver)"),
            ("Problem Solver (test)", "$(typeof(test_solver)) $(test_solver)"),
            ("Training Instances", string(num_train)),
            ("Testing Instances", string(num_test)),
        ]; caption="Data configuration for run $(run_id)", label="tab:data-$(label_base)")

        # ---- Model Configuration ----
        println(io, "\\subsubsection{Model Configuration}")
        write_latex_kv_table(io, [
            ("Method", string(typeof(model.method))),
            ("GNN Layers", string(model.gnn.config.num_layers)),
            ("GNN Hidden Dim", string(model.gnn.config.hidden_dim)),
            ("Total Parameters", string(num_params)),
            ("Timesteps (T)", string(model.method.scheduler.total_steps)),
            ("Inference Steps", string(model.method.scheduler.inference_steps)),
        ]; caption="Model configuration for run $(run_id)", label="tab:model-$(label_base)")

                # ---- Training Hyperparameters ----
        println(io, "\\subsubsection{Training Hyperparameters}")
        write_latex_kv_table(io, [
            ("Loaded pretrained model", string(model_already_exists)),
            ("Epochs", string(epochs)),
            ("Batch Size", string(batch_size)),
            ("Learning Rate", string(learning_rate)),
            ("Validation Instances", string(num_val)),
        ]; caption="Training hyperparameters for run $(run_id)", label="tab:hyperparams-$(label_base)")

        # ---- Results & Metrics ----
        println(io, "\\subsubsection{Results \\& Metrics}")
        results_rows = [
            ("Final Training Loss", model_already_exists ? "N/A (loaded)" : string(round(final_train_loss, digits=6))),
            ("Final Validation Loss", model_already_exists ? "N/A (loaded)" : (isnan(final_val_loss) ? "N/A (num_val=0)" : string(round(final_val_loss, digits=6)))),
            ("Avg Optimal Solution Size", string(round(avg_true_score, digits=2))),
            ("Avg Predicted Solution Size", string(round(avg_model_score, digits=2))),
            ("Approximation Ratio (\\%)", string(round(performance_pct, digits=2))),
            # B7: std/95% CI across the num_test instances of this run (within-run
            # variance — not cross-run/seed variance, which needs RNG-seeded
            # generators; see B6, not in scope for this fix).
            ("Approximation Ratio Std Dev (\\%)", isnan(approx_ratio_std) ? "N/A" : string(round(approx_ratio_std, digits=2))),
            ("Approximation Ratio 95\\% CI (\\%, ±)", isnan(approx_ratio_ci95) ? "N/A" : string(round(approx_ratio_ci95, digits=2))),
            ("Total Inference Time (s)", string(round(total_inference_time, digits=4))),
            ("Avg Inference Time / Instance (s)", string(round(avg_inference_time_per_instance, digits=4))),
        ]
        if !isnothing(ood_generator)
            append!(results_rows, [
                ("OOD Generator", string(typeof(ood_generator))),
                ("OOD Test Instances", string(ood_num_test)),
                ("OOD Approximation Ratio (\\%)", string(round(ood_performance_pct, digits=2))),
            ])
        end
        write_latex_kv_table(io, results_rows; caption="Results and performance metrics for run $(run_id)", label="tab:results-$(label_base)")

        # ---- Heatmap Quality ----
        println(io, "\\subsubsection{Heatmap Quality}")
        write_latex_kv_table(io, [
            ("Avg Pseudo-Entropy \$1 - \\bar{D}\$ (lower = more decisive)", string(round(avg_entropy, digits=4))),
            ("Avg Decisiveness Score \$\\bar{D} = mean(|2p-1|)\$ (higher = better)", string(round(avg_confidence, digits=4))),
            ("Avg Greedy Correction Rate", string(round(avg_correction_rate, digits=4))),
        ]; caption="Heatmap decisiveness (mean deviation from 0.5) and greedy-decoding correction metrics for run $(run_id)", label="tab:heatmap-$(label_base)")

        # ---- Inference Stability ----
        println(io, "\\subsubsection{Inference Stability}")
        write_latex_kv_table(io, [
            ("Repeated Inferences on Last Instance", string(num_repeat_inference)),
            ("Avg Normalized Hamming Distance across repeats", isnan(avg_hamming_stability) ? "N/A" : string(round(avg_hamming_stability, digits=4))),
        ]; caption="Repeated-inference stability metrics for run $(run_id)", label="tab:stability-$(label_base)")
    end

    @info "LaTeX report saved to $filepath"

    return filepath
end

"""
    latex_escape(s)

Escapes LaTeX special characters in a value before embedding it in generated
`.tex` output (run_ids, type names, etc. commonly contain `_`).

NOTE: this function still has the A14 double-escaping bug (backslash
substitution runs before the brace-escaping loop, so a literal backslash in
`s` would come out as `\\textbackslash\\{\\}` instead of
`\\textbackslash{}`). A14 was not in this round's requested fix list (B10,
B11, B12, C6, D4, D5, D6, B7), so it is left exactly as it was — flagging
here so it isn't mistaken for an oversight.
"""
function latex_escape(s)
    str = string(s)
    str = replace(str, "\\" => "\\textbackslash{}")
    for (from, to) in (
        "_" => "\\_", "%" => "\\%", "&" => "\\&", "#" => "\\#",
        "\$" => "\\\$", "{" => "\\{", "}" => "\\}",
        "^" => "\\textasciicircum{}", "~" => "\\textasciitilde{}",
    )
        str = replace(str, from => to)
    end
    return str
end

"""
    write_latex_kv_table(io, rows; caption=nothing, label=nothing)

Writes a simple two-column `Parameter | Value` LaTeX table.
"""
function write_latex_kv_table(io, rows::Vector{<:Tuple}; caption=nothing, label=nothing)
    println(io, "\\begin{table}[h!]")
    println(io, "\\centering")
    println(io, "\\begin{tabular}{ll}")
    println(io, "\\toprule")
    println(io, "\\textbf{Parameter} & \\textbf{Value} \\\\")
    println(io, "\\midrule")
    for (k, v) in rows
        println(io, "$(latex_escape(k)) & $(latex_escape(v)) \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    if !isnothing(caption)
        println(io, "\\caption{$(latex_escape(caption))}")
    end
    if !isnothing(label)
        println(io, "\\label{$label}")
    end
    println(io, "\\end{table}")
    println(io)
end
