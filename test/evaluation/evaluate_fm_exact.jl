using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CUDA
using cuDNN
using Flux
using Random
using DiFoG

include(joinpath(pkgdir(DiFoG), "test", "test_utils.jl"))

# H1 test: does the Euler step rule of discrete flow matching hurt, because it commits fewer nodes per step
# than the mixture path the model was trained on (κ(τ) = τ² has κ̇(0) = 0, so the first Euler step commits
# nothing)? The already TRAINED Euler flow-matching models of evaluate.jl are evaluated again with the EXACT
# step rule c = (κ(τ') − κ(τ)) / (1 − κ(τ)). Nothing is trained: weights, test instances, batch size and the
# ATSS grids are the same as in the Euler runs, only the sampler differs. The 1-step cells are a control:
# with a single step both rules sample directly from the prediction, so they must match the Euler reports.

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
