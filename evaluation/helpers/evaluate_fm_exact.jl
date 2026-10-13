using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CUDA
using cuDNN
using Flux
using Random
using DiFoG

include(joinpath(@__DIR__, "..", "pipeline.jl"))

# H1 test: does the Euler step rule of discrete flow matching hurt, because it commits fewer nodes per step
# than the mixture path the model was trained on (κ(τ) = τ² has κ̇(0) = 0, so the first Euler step commits
# nothing)? The already TRAINED Euler flow-matching models of evaluate.jl are evaluated again with the EXACT
# step rule c = (κ(τ') − κ(τ)) / (1 − κ(τ)). Nothing is trained: weights, test instances, batch size and the
# ATSS grids are the same as in the Euler runs, only the sampler differs. The 1-step cells are a control:
# with a single step both rules sample directly from the prediction, so they must match the Euler reports.
#
# Results (ER-50-100, Gurobi labels, 1000 test instances; mean optimality gap of the Euler-trained models sampled
# with the Euler and with the exact step rule, Δ = exact − Euler in percentage points, negative = exact better).
# The exact rule helps most at 2 steps (up to −4.3 pp for MISP × ATSS), where the Euler rule commits too few nodes
# in the first step; with many steps both rules converge. The 1-step cells agree up to sampling noise (control).
# This table is Table "Step rule" in the evaluation chapter; the thesis results use the exact rule.
#
#                     |          MISP            |         MaxCut
#   Schedule   Steps  |  Euler   Exact      Δ    |  Euler   Exact      Δ
#   Linear         1  |  4.76%   4.81%   +0.05   |  3.67%   3.69%   +0.02
#                  2  |  6.17%   4.68%   −1.49   |  3.20%   3.52%   +0.32
#                  5  |  4.03%   3.96%   −0.07   |  2.66%   2.33%   −0.33
#                 10  |  3.43%   3.19%   −0.24   |  2.22%   2.08%   −0.14
#                 20  |  3.10%   3.07%   −0.03   |  2.08%   2.03%   −0.05
#                 50  |  2.76%   2.68%   −0.08   |  2.10%   2.05%   −0.05
#   Cosine         1  |  4.69%   4.71%   +0.02   |  3.55%   3.57%   +0.02
#                  2  |  7.07%   4.06%   −3.01   |  3.29%   2.03%   −1.26
#                  5  |  3.71%   3.19%   −0.52   |  1.36%   0.98%   −0.38
#                 10  |  2.85%   2.80%   −0.05   |  0.92%   0.95%   +0.03
#                 20  |  2.43%   2.45%   +0.02   |  0.97%   1.08%   +0.11
#                 50  |  2.37%   2.22%   −0.15   |  3.97%   2.26%   −1.71
#   ATSS           1  |  4.87%   4.62%   −0.25   |  3.58%   3.68%   +0.10
#                  2  |  7.60%   3.33%   −4.27   |  3.22%   1.02%   −2.20
#                  5  |  4.39%   2.81%   −1.58   |  1.64%   1.29%   −0.35
#                 10  |  4.36%   2.63%   −1.73   |  4.17%   4.20%   +0.03
#                 20  |  3.19%   2.39%   −0.80   |  3.95%   3.90%   −0.05
#                 50  |  2.99%   2.59%   −0.40   |  0.98%   1.09%   +0.11

const METHOD_NAME = "flow_matching_exact_inf"   # report name; must differ from the trained method
const TRAINED_METHOD = "flow_matching"          # Euler-trained model whose weights are loaded

"""
    build_run_id(problem_name, method_name, generator, gnn_config, scheduler_name, inference_steps)

Same format as in evaluate.jl, e.g. "misp_flow_matching_exact_inf_er_50_100_12_256_linear_2".
"""
function build_run_id(problem_name::String, method_name::String, generator::ERGraphGenerator,
                       gnn_config::GNNConfig, scheduler_name::String, inference_steps::Int)
    return "$(problem_name)_$(method_name)_er_$(generator.min_nodes)_$(generator.max_nodes)_" *
           "$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_$(scheduler_name)_$(inference_steps)"
end

"""
    build_train_run_id(problem_name, method_name, generator, gnn_config)

Same format as in evaluate.jl; identifies the saved model file.
"""
function build_train_run_id(problem_name::String, method_name::String, generator::ERGraphGenerator,
                             gnn_config::GNNConfig)
    return "$(problem_name)_$(method_name)_er_$(generator.min_nodes)_$(generator.max_nodes)_" *
           "$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_train"
end

"""
    build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p)

Constructs the requested `TimeSchedule`. `generator`/`solver` are only used by "atss".
"""
function build_scheduler(scheduler_name::String, generator::GraphGenerator, solver::ProblemSolver,
                          total_steps::Int, inference_steps::Int, κ::Function, κ̇::Function, atss_p::Float32)
    if scheduler_name == "linear"
        return LinearScheduler(total_steps, inference_steps)
    elseif scheduler_name == "cosine"
        return CosineScheduler(total_steps, inference_steps)
    elseif scheduler_name == "atss"
        return ATSSScheduler(generator, solver, total_steps, inference_steps, κ, κ̇; atss_p=atss_p)
    else
        error("Unknown scheduler_name: $scheduler_name")
    end
end

"""
    execute_case(; ...)

Evaluates one cell with the exact sampler on the loaded Euler-trained model. The ATSS fit is seeded with the
run_id of the corresponding EULER cell, so both samplers use the identical ATSS grid (the fit simulates the
mixture path and does not depend on the sampler).
"""
function execute_case(; run_id::String, euler_run_id::String, train_run_id::String, scheduler_name::String,
                       inference_steps::Int, generator::ERGraphGenerator, gnn_config::GNNConfig,
                       total_steps::Int, κ::Function, κ̇::Function, atss_p::Float32, λ_cons::Float32,
                       result_root::String, training_seed::Int, test_seed::Int, validation_seed::Int,
                       num_train::Int, num_test::Int, epochs::Int, batch_size::Int, learning_rate::Float64,
                       num_repeat_inference::Int, solver::ProblemSolver)

    if scheduler_name == "atss"
        seed_everything!("atss:" * euler_run_id)
    end
    scheduler = build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p)
    method = DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇; sampler=:exact)

    seed_everything!("init:" * train_run_id)   # irrelevant (weights are loaded), kept for symmetry
    model = DiffusionModel(AGNN(gnn_config), method) |> Flux.gpu

    return run_automated_pipeline(;
        run_id, train_run_id, model, generator, solver, num_train, num_test, epochs, batch_size,
        learning_rate, num_repeat_inference, evaluate = true, result_root, training_seed, test_seed,
        validation_seed,
    )
end

function main()
    result_root = "/home/e12023149/thesis/results"

    # ---- Must match evaluate.jl, otherwise the comparison with the Euler reports is not valid ----
    generator = ERGraphGenerator(min_nodes=50, max_nodes=100, p=0.15)
    solver_time_limit = 60.0
    gnn_config = GNNConfig(num_layers=12, in_dim=1, time_dim=32, hidden_dim=256, output_dim=1)
    total_steps = 1000
    κ  = t -> t^2.0f0
    κ̇  = t -> 2.0f0 * t^1.0f0
    atss_p = 0.5f0
    λ_cons = 0.1f0                  # only enters the consistency loss (training); reported for completeness

    num_train = 10000               # only used to locate/report the training set; nothing is trained
    num_test = 1000
    epochs = 5                      # only written to the report; set to the epochs the FM models were trained with
    batch_size = 32                 # must match: BatchNorm normalizes over the evaluation batch
    learning_rate = 0.001
    num_repeat_inference = 100

    seeds = Dict(
        "misp"   => (training = 5010001, test = 5010002, validation = 5010003),
        "maxcut" => (training = 5010004, test = 5010005, validation = 5010006),
    )

    # ---- The grid: MISP is the question; add "maxcut" as a second control if wanted ----
    problems             = ["misp"]
    schedulers           = ["linear", "cosine", "atss"]
    inference_steps_list = [1, 2, 5, 10, 20, 50]   # 1 step = control (both samplers identical)
    skip_existing_reports = true

    common = (; generator, gnn_config, total_steps, κ, κ̇, atss_p, λ_cons, result_root, num_train, num_test,
                epochs, batch_size, learning_rate, num_repeat_inference)

    for problem_name in problems
        solver = problem_name == "misp" ? MISPSolver(time_limit=solver_time_limit) :
                                           MaxCutSolver(time_limit=solver_time_limit)
        s = seeds[problem_name]

        # Never train here: without the Euler-trained model, run_automated_pipeline would train a new one
        # under the flow_matching train_run_id, with the exact rule inside the consistency loss.
        train_id = build_train_run_id(problem_name, TRAINED_METHOD, generator, gnn_config)
        model_path = joinpath(result_root, "models", "model-$train_id.jld2")
        if !isfile(model_path)
            @error "Euler-trained model $model_path not found — run evaluate.jl first. Skipping $problem_name."
            continue
        end

        for scheduler_name in schedulers, inference_steps in inference_steps_list
            run_id = build_run_id(problem_name, METHOD_NAME, generator, gnn_config, scheduler_name, inference_steps)
            euler_run_id = build_run_id(problem_name, TRAINED_METHOD, generator, gnn_config, scheduler_name, inference_steps)
            if skip_existing_reports && isfile(joinpath(result_root, "reports", "report-$run_id.tex"))
                @info "Skipping $run_id (report already exists)."
                continue
            end

            @info "=== Evaluating $run_id (weights: $train_id, exact sampler) ==="
            try
                execute_case(; common..., run_id, euler_run_id, train_run_id = train_id, scheduler_name,
                             inference_steps, solver, training_seed = s.training, test_seed = s.test,
                             validation_seed = s.validation)
            catch e
                @error "Evaluation $run_id FAILED — continuing with the next cell." exception=(e, catch_backtrace())
            end
        end
    end

    @info "Exact-sampler grid complete."
end

Base.invokelatest(main)
