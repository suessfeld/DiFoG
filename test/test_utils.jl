using Flux
using GraphNeuralNetworks
using Printf
using Dates
using Statistics
using Random
using Base.Iterators: partition

"""
    stable_seed(s::AbstractString)

Deterministic 32-bit FNV-1a hash of `s`, used as RNG seed (R31). Unlike `hash`, it does not depend on
the Julia version, so seeds derived from run ids are reproducible across sessions and machines.
"""
function stable_seed(s::AbstractString)
    h = UInt32(2166136261)
    for b in codeunits(s)
        h = (h ⊻ UInt32(b)) * UInt32(16777619)
    end
    return Int(h)
end

"""
    seed_everything!(key::AbstractString)

Seeds the global CPU RNG and (if functional) the CUDA RNG from `stable_seed(key)`.
"""
function seed_everything!(key::AbstractString)
    seed = stable_seed(key)
    Random.seed!(seed)
    DiFoG.CUDA.functional() && DiFoG.CUDA.seed!(seed)
    return seed
end

_finite(v) = filter(isfinite, v)
_mean_or_nan(v) = isempty(v) ? NaN : mean(v)
_fmt(x; digits=4) = isnan(x) ? "N/A" : string(round(x, digits=digits))

"""
    _instance_gap(g::GNNGraph)

Relative gap of the reference solution of `g` as stored by `RuntimeDataset` (`gdata.gap`), NaN if absent.
"""
_instance_gap(g::GNNGraph) = :gap in keys(g.gdata) ? Float64(g.gdata.gap[1]) : NaN

"""
    _ratio_stats(model_scores, ref_scores)

Per-instance ratios 100·model/reference (instances with reference 0 are skipped) → `(mean, std, ci95)`.
The headline number and its CI are computed from the SAME quantity (H5).
"""
function _ratio_stats(model_scores::AbstractVector, ref_scores::AbstractVector)
    r = [100 * m / t for (m, t) in zip(model_scores, ref_scores) if t > 0]
    μ = _mean_or_nan(r)
    σ = length(r) > 1 ? std(r) : NaN
    return μ, σ, length(r) > 1 ? 1.96 * σ / sqrt(length(r)) : NaN
end

"""
    _method_label(method)

Descriptive method name for reports (M5): the flow-matching variant implemented here is DFM with the
reference-form consistency loss, not the full EFLOCO method (no ρ(t) training density).
"""
_method_label(m::BernoulliDiffusion) = "Bernoulli diffusion (DIFUSCO)"
_method_label(m::DiscreteFlowMatching) =
    "DFM + consistency (reference form, lambda_cons=$(m.λ_cons)), sampler=$(m.sampler)"
_method_label(m::RectifiedFlow) =
    "Rectified flow (Gaussian relaxation, source=$(source_label(m.source))), prediction=$(m.prediction), Euler ODE"

"""
    _model_eval_signal(method)

Signal level at every time the network is evaluated during sampling (H2), in evaluation order, formatted
identically for all methods as "time: signal":
ᾱ_t for Bernoulli diffusion (probability of keeping x_0 is (1+ᾱ)/2; ᾱ ≈ 0 means pure noise),
κ_τ (fraction committed to the target) for flow matching,
τ (weight of the target on the straight path x_τ = τ x₁ + (1 − τ) x₀) for rectified flow.
"""
function _model_eval_signal(m::BernoulliDiffusion)
    ts = reverse(m.scheduler.steps)[1:end-1]            # T → … → smallest non-zero step
    return join(["t=$(t): $(round(m.ᾱ[t], sigdigits=2))" for t in ts], ", ")
end
# For :euler this is the NOMINAL κ(τ) of the training path; the Euler sampler under-commits (H1), so the
# state the model actually receives is less committed (at τ = 0 nothing is committed).
function _model_eval_signal(m::DiscreteFlowMatching)
    τs = m.scheduler.steps[1:end-1]                     # 0 → … → last step before 1
    return join(["tau=$(round(τ, digits=3)): $(round(Float64(m.κ(τ)), sigdigits=2))" for τ in τs], ", ")
end
# Straight path: the target weight at time τ is τ itself.
function _model_eval_signal(m::RectifiedFlow)
    τs = m.scheduler.steps[1:end-1]
    return join(["tau=$(round(τ, digits=3)): $(round(Float64(τ), sigdigits=2))" for τ in τs], ", ")
end

"""
    run_automated_pipeline(; run_id, train_run_id=run_id, model, generator, solver, num_train=1000,
                           num_test=100, num_val=max(20, div(num_train, 20)), epochs=5, batch_size=16,
                           learning_rate=1e-3, num_repeat_inference=5, evaluate=true,
                           result_root, training_seed, test_seed, validation_seed)

Runs training (or loads an existing model) and, unless `evaluate=false`, a full evaluation that is
written as a LaTeX report.

Parameters:
    - 'run_id': identifies this run's report, instance dumps (defaults to a timestamp).
    - 'train_run_id': identifies the trained model file (`models/model-<train_run_id>.jld2`) and its loss
      curve. Defaults to `run_id`; pass a different value to share one trained model across many
      evaluation cells (scheduler type and `inference_steps` are only used at inference).
    - 'model': The model that should be trained and tested.
    - 'generator': The graph generator.
    - 'solver': The problem solver (labels, decoder, objective, greedy baseline).
    - 'num_train', 'num_test', 'num_val': dataset sizes (`num_val = 0` disables validation).
    - 'epochs', 'batch_size', 'learning_rate': training hyperparameters.
    - 'num_repeat_inference': repeated inferences on the last test instance (stability measurement).
    - 'evaluate': `false` → only train/load and save the model, no test evaluation or report (R32).
    - 'result_root': Root directory for models/graphs/seeds/reports/instances.
    - 'training_seed', 'test_seed', 'validation_seed': seeds identifying each split's `<seed>.csv`.
    - 'reflow_from': `RectifiedFlow` only. `train_run_id` of a trained (k-)RF model: its weights are loaded,
      its ODE coupling (x₀, Z₁) is generated on the training/validation instances, and the model is
      trained on that coupling instead of the data labels ((k+1)-RF, Reflow). `nothing` = regular training.
    - 'reflow_ode_steps': Euler steps used to generate the Reflow coupling.

Returns the report path, or `nothing` when `evaluate=false`.
"""
function run_automated_pipeline(;
    run_id::String = Dates.format(Dates.now(), "yyyymmdd-HHMMSS"),
    train_run_id::String = run_id,
    model::DiffusionModel,
    generator::GraphGenerator,
    solver::ProblemSolver,
    num_train::Int = 1000,
    num_test::Int = 100,
    num_val::Int = max(20, div(num_train, 20)),
    epochs::Int = 5,
    batch_size::Int = 16,
    learning_rate::Float64 = 1e-3,
    num_repeat_inference::Int = 5,
    evaluate::Bool = true,
    result_root::String,
    training_seed::Int,
    test_seed::Int,
    validation_seed::Int,
    reflow_from::Union{Nothing,String} = nothing,
    reflow_ode_steps::Int = 100
)
    @info "=== Starting Pipeline Run: $run_id (trained model: $train_run_id, evaluate=$evaluate) ==="
    seeds_dir = joinpath(result_root, "seeds")
    symmetric = DiFoG.label_symmetric(solver)

    models_dir = joinpath(result_root, "models")
    mkpath(models_dir)
    model_path = joinpath(models_dir, "model-$train_run_id.jld2")

    model_already_exists = isfile(model_path)

    final_train_loss = NaN
    final_val_loss = NaN
    val_loader = nothing

    if model_already_exists
        @info "Found existing model for train_run_id '$train_run_id' at $model_path — loading weights and skipping training."
        DiFoG.load_model!(model, model_path)
    else
        @info "No existing model found for train_run_id '$train_run_id' — training from scratch."

        # R31: training randomness (t sampling, corruption noise) depends only on train_run_id,
        # not on how many cells ran before in the same session.
        seed_everything!("train:" * train_run_id)

        val_loader = nothing
        if num_val > 0
            @info "Building fixed validation set ($(num_val) instances, held constant across epochs)..."
            val_dataset = RuntimeDataset(num_val, generator, solver, validation_seed, seeds_dir)
            val_instances = [val_dataset[i] for i in 1:num_val]
            val_loader = Flux.DataLoader(val_instances, batchsize=batch_size, shuffle=false, collate=true)
        else
            @info "num_val=0 — validation loss tracking disabled for this run."
        end

        @info "Initializing Train dataset..."
        train_dataset = RuntimeDataset(num_train, generator, solver, training_seed, seeds_dir)
        train_loader = Flux.DataLoader(train_dataset, batchsize=batch_size, shuffle=false, collate=true)

        if !isnothing(reflow_from)
            model.method isa RectifiedFlow || error("reflow_from is only supported for RectifiedFlow, got $(typeof(model.method))")
            teacher_path = joinpath(models_dir, "model-$reflow_from.jld2")
            isfile(teacher_path) || error("Reflow teacher model '$reflow_from' not found at $teacher_path — train it first.")
            @info "Reflow: warm-starting from $teacher_path and generating its coupling ($(reflow_ode_steps) Euler steps)..."
            DiFoG.load_model!(model, teacher_path)
            coupling = DiFoG.generate_reflow_coupling(model, train_loader; ode_steps=reflow_ode_steps)
            train_loader = Flux.DataLoader(coupling, batchsize=batch_size, shuffle=true, collate=true)
            if !isnothing(val_loader)
                val_coupling = DiFoG.generate_reflow_coupling(model, val_loader; ode_steps=reflow_ode_steps)
                val_loader = Flux.DataLoader(val_coupling, batchsize=batch_size, shuffle=false, collate=true)
            end
        end

        optim = Flux.setup(
            Flux.Optimisers.OptimiserChain(
                Flux.Optimisers.ClipNorm(1.0f0),   # M2: norm clipping (ClipGrad clips each element), per array
                Flux.Adam(learning_rate)
            ),
            model.gnn
        )

        @info "Training model for $epochs epochs (symmetric labels: $symmetric)..."
        loss_history, val_loss_history, batch_loss_history, val_loss_batches =
            DiFoG.train_diffusion_model!(model, optim, train_loader; epochs=epochs, val_loader=val_loader,
                                          symmetric_labels=symmetric)
        final_train_loss = isempty(loss_history) ? NaN : loss_history[end]
        final_val_loss = isempty(val_loss_history) ? NaN : val_loss_history[end]

        @info "Saving trained model weights immediately after training completes..."
        DiFoG.save_model(model, model_path)
        @info "Model successfully saved to $model_path"

        graphs_dir = joinpath(result_root, "graphs")
        mkpath(graphs_dir)
        loss_plot_path = joinpath(graphs_dir, "loss-$train_run_id.png")
        DiFoG.plot_loss_curve(batch_loss_history, val_loss_history, val_loss_batches, loss_plot_path, batch_size;
                               title="Training Loss — $train_run_id")
        @info "Loss curve saved to $loss_plot_path"
    end

    if !evaluate
        @info "evaluate=false — skipping test evaluation for $run_id."
        return nothing
    end

    # Evaluation randomness (sampler noise) depends only on run_id.
    seed_everything!("eval:" * run_id)

    num_params = sum(length, Flux.trainables(model.gnn))

    @info "Initializing Test dataset..."
    test_dataset = RuntimeDataset(num_test, generator, solver, test_seed, seeds_dir)
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
    per_instance_greedy_scores = Float64[]
    instance_gaps = Float64[]

    last_test_graph = nothing
    last_test_instance_idx = 0

    test_results_dir = joinpath(result_root, "instances", "test_instances_$run_id")
    mkpath(test_results_dir)

    Flux.testmode!(model.gnn)

    global_test_idx = 1
    batch_num = 0

    for batch in test_loader
        graphs = [item[1] for item in batch]
        names = ["test_instance_$(global_test_idx + k - 1).txt" for k in 1:length(graphs)]
        global_test_idx += length(graphs)
        batch_num += 1

        g_batch = Flux.batch(graphs)
        g_batch_gpu = g_batch |> Flux.gpu
        target_x0 = g_batch_gpu.ndata.x

        shape = size(target_x0)
        batch_size_actual = length(graphs)

        batch_inference_time = @elapsed begin
            preds_probs = DiFoG.sample_heatmap(model.method, model, shape, g_batch_gpu)
            DiFoG._sync_device()
        end
        total_inference_time += batch_inference_time
        avg_time_per_instance = batch_inference_time / batch_size_actual

        @info "Batch $(batch_num) [$(total_graphs + batch_size_actual) instances total] | Total: $(round(batch_inference_time, digits=4))s | Per instance: $(round(avg_time_per_instance, digits=4))s"

        true_mask_cpu = Flux.cpu(target_x0) .> 0.5f0
        preds_probs_cpu = Flux.cpu(preds_probs)

        valid_masks_cpu = Vector{Vector{Int}}()
        edge_data = Vector{Tuple{Vector{Int32},Vector{Int32}}}()
        node_masks = Vector{Vector{Int}}()
        batch_model_score = 0.0
        batch_true_score = 0.0

        start_slice = 1
        for j in 1:length(graphs)
            num_nodes = graphs[j].num_nodes
            end_slice = start_slice + num_nodes - 1

            heatmap_slice = preds_probs_cpu[1, start_slice:end_slice]
            heatmap_vec = vec(heatmap_slice)
            true_vec = vec(true_mask_cpu[1, start_slice:end_slice])
            mask_slice = DiFoG.decode_heatmap(solver, graphs[j], heatmap_vec)
            push!(valid_masks_cpu, mask_slice)

            s_j, t_j = edge_index(graphs[j])
            push!(edge_data, (copy(s_j), copy(t_j)))
            push!(node_masks, Int.(vec(graphs[j].ndata.x)))

            model_score_j = DiFoG.evaluate(solver, graphs[j], mask_slice)
            true_score_j  = DiFoG.evaluate(solver, graphs[j], true_vec)
            batch_model_score += model_score_j
            batch_true_score  += true_score_j
            push!(per_instance_model_scores, model_score_j)
            push!(per_instance_true_scores, true_score_j)

            # Q16: greedy baseline through the same decoder (MIS: min-degree greedy, MaxCut: degree-ordered greedy cut)
            greedy_mask = DiFoG.decode_heatmap(solver, graphs[j], DiFoG.degree_baseline_heatmap(solver, graphs[j]))
            push!(per_instance_greedy_scores, DiFoG.evaluate(solver, graphs[j], greedy_mask))

            # R28: quality of the reference solution itself
            push!(instance_gaps, _instance_gap(graphs[j]))

            mean_entropy, confidence = DiFoG.heatmap_decisiveness(heatmap_vec)
            total_entropy += mean_entropy
            total_confidence += confidence

            corr_dist = DiFoG.correction_distance(heatmap_vec, mask_slice)
            total_correction_rate += corr_dist / num_nodes

            start_slice = end_slice + 1
        end

        total_model_score += batch_model_score
        total_true_score += batch_true_score
        total_graphs += batch_size_actual

        last_test_graph = graphs[end]
        last_test_instance_idx = global_test_idx - 1

        # File writing runs synchronously (not Threads.@spawn'd): GraphNeuralNetworks.jl's edge storage
        # was confirmed to mutate under concurrent access.
        start_idx = 1
        for j in 1:length(graphs)
            num_nodes = graphs[j].num_nodes
            end_idx = start_idx + num_nodes - 1

            heatmap_slice = preds_probs_cpu[1, start_idx:end_idx]
            mask_slice = valid_masks_cpu[j]
            s, t = edge_data[j]
            num_edges = div(length(s), 2)

            filepath = joinpath(test_results_dir, replace(names[j], ".txt" => "_result.txt"))

            open(filepath, "w") do io
                println(io, "$num_nodes $num_edges")
                for k in 1:length(s)
                    if s[k] < t[k]
                        println(io, "$(s[k]) $(t[k])")
                    end
                end
                println(io, join(node_masks[j], " "))
                println(io, join(Int.(vec(mask_slice)), " "))
                println(io, join([@sprintf("%.4f", x) for x in vec(heatmap_slice)], " "))
            end

            start_idx = end_idx + 1
        end
    end

    # generic evaluation parameters
    avg_model_score = total_model_score / total_graphs
    avg_true_score = total_true_score / total_graphs
    ratio_of_means_pct = (avg_model_score / avg_true_score) * 100          # secondary: ratio of averages
    # H5: headline = mean of per-instance ratios, with std and CI of the same quantity
    performance_pct, approx_ratio_std, approx_ratio_ci95 = _ratio_stats(per_instance_model_scores, per_instance_true_scores)
    greedy_pct, _, greedy_ci95 = _ratio_stats(per_instance_greedy_scores, per_instance_true_scores)
    # absolute objective values, mean ± σ over instances (as reported in the literature, e.g. Bresich et al.)
    _obj_stats(v) = "$(_fmt(_mean_or_nan(v); digits=2)) ± $(_fmt(length(v) > 1 ? std(v) : NaN; digits=2))"
    avg_inference_time_per_instance = total_inference_time / total_graphs

    avg_confidence = total_confidence / total_graphs
    avg_entropy = total_entropy / total_graphs
    avg_correction_rate = total_correction_rate / total_graphs
    corr_rate_str = solver isa MaxCutSolver ? "N/A (threshold decoder)" : string(round(avg_correction_rate, digits=4))

    finite_gaps = _finite(instance_gaps)
    n_gap_missing = length(instance_gaps) - length(finite_gaps)
    # Gurobi's default MIPGap: OPTIMAL status at relative gap ≤ 1e-4
    frac_proven_optimal = isempty(finite_gaps) ? NaN : count(<=(1e-4), finite_gaps) / length(finite_gaps)
    mean_gap = _mean_or_nan(finite_gaps)
    max_gap = isempty(finite_gaps) ? NaN : maximum(finite_gaps)


    @info "Testing Complete. Avg Model Score: $(round(avg_model_score, digits=2)) | Avg Reference Score: $(round(avg_true_score, digits=2)) | Mean per-instance ratio vs. reference: $(round(performance_pct, digits=2))% ± $(round(approx_ratio_ci95, digits=2))% (95% CI) | Greedy baseline: $(round(greedy_pct, digits=2))% | Avg Inference Time: $(round(avg_inference_time_per_instance, digits=4))s/instance"
    @info "Heatmap Quality | Avg Decisiveness: $(round(avg_confidence, digits=4)) (H̄=$(round(avg_entropy, digits=4))) | Avg Greedy Correction Rate: $(round(avg_correction_rate, digits=4))"
    @info "Reference solutions | optimal within MIPGap 1e-4: $(_fmt(frac_proven_optimal)) | mean gap: $(_fmt(mean_gap; digits=6)) | max gap: $(_fmt(max_gap; digits=6)) | without solver result: $n_gap_missing"

    avg_hamming_stability = NaN
    if num_repeat_inference > 1 && !isnothing(last_test_graph)
        @info "Running $(num_repeat_inference) repeated inferences on instance $(last_test_instance_idx) for stability measurement..."

        g_repeat_gpu = last_test_graph |> Flux.gpu
        shape = (model.gnn.config.in_dim, g_repeat_gpu.num_nodes)   # M7

        s, t = edge_index(last_test_graph)
        num_edges = div(length(s), 2)
        gt_solution = Int.(vec(last_test_graph.ndata.x))

        repeat_masks = Vector{Vector{Int}}(undef, num_repeat_inference)

        for r in 1:num_repeat_inference
            heatmap_r = DiFoG.sample_heatmap(model.method, model, shape, g_repeat_gpu)
            heatmap_r_cpu = vec(Flux.cpu(heatmap_r))
            mask_r = Int.(vec(DiFoG.decode_heatmap(solver, last_test_graph, heatmap_r_cpu)))
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
            hd = DiFoG.solution_distance(repeat_masks[i], repeat_masks[j]; symmetric=symmetric)   # H5: min(d, n−d) for MaxCut
            push!(pairwise_distances, hd / last_test_graph.num_nodes)
        end
        avg_hamming_stability = isempty(pairwise_distances) ? 0.0 : mean(pairwise_distances)

        @info "Repeated Inference Stability | Instance $(last_test_instance_idx) | Avg Normalized Hamming Distance across $(num_repeat_inference) runs: $(round(avg_hamming_stability, digits=4))"
    end

    Flux.trainmode!(model.gnn)

    reports_dir = joinpath(result_root, "reports")
    mkpath(reports_dir)
    filepath = joinpath(reports_dir, "report-$run_id.tex")

    label_base = replace(run_id, r"[^A-Za-z0-9]" => "-")

    open(filepath, "w") do io
        println(io, "% Auto-generated pipeline report — copy directly into thesis .tex source")
        println(io, "\\subsection{Pipeline Run: $(latex_escape(run_id))}")
        println(io, "\\label{sec:run-$(label_base)}")
        println(io)

        println(io, "\\subsubsection{Data Configuration}")
        write_latex_kv_table(io, [
            ("Graph Generator", string(typeof(generator))),
            ("Generator Params", string(generator)),
            ("Problem Solver", "$(typeof(solver)) $(solver)"),
            ("Training Instances", string(num_train)),
            ("Testing Instances", string(num_test)),
            ("Reference Solutions Optimal within MIPGap 1e-4 (fraction)", _fmt(frac_proven_optimal)),
            ("Reference Solution Mean Gap", _fmt(mean_gap; digits=6)),
            ("Reference Solution Max Gap", _fmt(max_gap; digits=6)),
            ("Instances Without Solver Result", string(n_gap_missing)),
        ]; caption="Data configuration for run $(run_id)", label="tab:data-$(label_base)")

        println(io, "\\subsubsection{Model Configuration}")
        write_latex_kv_table(io, [
            ("Method", _method_label(model.method)),
            ("GNN Layers", string(model.gnn.config.num_layers)),
            ("GNN Hidden Dim", string(model.gnn.config.hidden_dim)),
            ("Time Embedding Dim", string(model.gnn.config.time_dim)),
            ("Total Parameters", string(num_params)),
            ("Timesteps (T)", string(model.method.scheduler.total_steps)),
            ("Inference Schedule", string(typeof(model.method.scheduler).name.name)),
            ("Inference Steps", string(model.method.scheduler.inference_steps)),
            ("Configured / effective transitions", "$(model.method.scheduler.inference_steps) / $(length(model.method.scheduler.steps) - 1)"),
            ("Signal level at model evaluations (alpha-bar / kappa / tau)", _model_eval_signal(model.method)),
        ]; caption="Model configuration for run $(run_id)", label="tab:model-$(label_base)")

        println(io, "\\subsubsection{Training Hyperparameters}")
        write_latex_kv_table(io, [
            ("Trained Model", train_run_id),
            ("Reflow Teacher Model", isnothing(reflow_from) ? "none" : "$reflow_from ($reflow_ode_steps Euler steps)"),
            ("Loaded pretrained model", string(model_already_exists)),
            ("Epochs", string(epochs)),
            ("Batch Size", string(batch_size)),
            ("Learning Rate", string(learning_rate)),
            ("Validation Instances", string(num_val)),
            ("Flip-invariant Loss", string(symmetric)),
        ]; caption="Training hyperparameters for run $(run_id)", label="tab:hyperparams-$(label_base)")

        println(io, "\\subsubsection{Results \\& Metrics}")
        # keys use a plain "%": latex_escape turns it into "\\%" (pre-escaped keys would be double-escaped)
        results_rows = [
            ("Final Training Loss", model_already_exists ? "N/A (loaded)" : string(round(final_train_loss, digits=6))),
            ("Final Validation Loss", model_already_exists ? "N/A (loaded)" : (isnan(final_val_loss) ? "N/A (num_val=0)" : string(round(final_val_loss, digits=6)))),
            ("Avg Reference Solution Score", string(round(avg_true_score, digits=2))),
            ("Avg Predicted Solution Score", string(round(avg_model_score, digits=2))),
            ("Predicted Objective, Mean ± Std", _obj_stats(per_instance_model_scores)),
            ("Reference Objective, Mean ± Std", _obj_stats(per_instance_true_scores)),
            ("Greedy Objective, Mean ± Std", _obj_stats(per_instance_greedy_scores)),
            ("Mean Per-Instance Ratio vs. Reference (%)", _fmt(performance_pct; digits=2)),
            ("Per-Instance Ratio Std Dev (%)", _fmt(approx_ratio_std; digits=2)),
            ("Per-Instance Ratio 95% CI (%, ±)", _fmt(approx_ratio_ci95; digits=2)),
            ("Ratio of Average Scores (%)", _fmt(ratio_of_means_pct; digits=2)),
            ("Greedy Baseline, Mean Ratio (%)", "$(_fmt(greedy_pct; digits=2)) ± $(_fmt(greedy_ci95; digits=2))"),
            ("Total Inference Time (s)", string(round(total_inference_time, digits=4))),
            ("Avg Inference Time / Instance (s)", string(round(avg_inference_time_per_instance, digits=4))),
        ]
        write_latex_kv_table(io, results_rows; caption="Results and performance metrics for run $(run_id)", label="tab:results-$(label_base)")

        println(io, "\\subsubsection{Heatmap Quality}")
        write_latex_kv_table(io, [
            ("Avg Pseudo-Entropy 1 - D (lower = more decisive)", string(round(avg_entropy, digits=4))),
            ("Avg Decisiveness Score D = mean(|2p-1|) (higher = better)", string(round(avg_confidence, digits=4))),
            ("Avg Greedy Correction Rate", corr_rate_str),
        ]; caption="Heatmap decisiveness and greedy-decoding correction metrics for run $(run_id)", label="tab:heatmap-$(label_base)")

        println(io, "\\subsubsection{Inference Stability}")
        write_latex_kv_table(io, [
            ("Repeated Inferences on Last Instance", string(num_repeat_inference)),
            ("Avg Normalized Solution Distance across repeats" * (symmetric ? " (complement-invariant)" : ""), _fmt(avg_hamming_stability)),
        ]; caption="Repeated-inference stability metrics for run $(run_id). Measured at batch size 1; BatchNorm normalizes over the whole batch (also in test mode), so heatmaps in the main evaluation (batch size $(batch_size)) also depend on the co-batched graphs", label="tab:stability-$(label_base)")
    end

    @info "LaTeX report saved to $filepath"

    return filepath
end

"""
    latex_escape(s)

Escapes LaTeX special characters in a value before embedding it in generated `.tex` output.

NOTE: still has the A14 double-escaping bug (backslash substitution runs before the brace-escaping
loop). Not in scope of this round; left as is and flagged so it isn't mistaken for an oversight.
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
    println(io, "\\begin{tabular}{p{0.42\\linewidth}p{0.5\\linewidth}}")   # p-columns wrap long values
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