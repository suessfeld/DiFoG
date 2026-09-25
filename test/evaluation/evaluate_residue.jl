using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CUDA
using cuDNN
using Flux
using Random
using DiFoG

include(joinpath(pkgdir(DiFoG), "test", "test_utils.jl"))

# Rectified Flow with a residue source (DISCO, Zhao et al. 2025): generation starts from a degraded solution
# plus noise, x₀ = signal(X_d) + σ ε, instead of pure Gaussian noise. Two degraded solutions are compared:
#   C1 "rectified_flow_residue_random":    uniformly random node subset, as DISCO uses for the MIS,
#   C2 "rectified_flow_residue_heuristic": the problem's `residue` (degree greedy for the MISP).
# Everything else matches the "rectified_flow" runs of evaluate.jl, so they can be compared cell by cell.

"""
    build_run_id(problem_name, method_name, generator, gnn_config, scheduler_name, inference_steps)

Same format as in evaluate.jl, e.g. "misp_rectified_flow_residue_random_er_50_100_12_256_linear_2".
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
    build_method(method_name, scheduler, solver, residue_σ)

- "rectified_flow_residue_random":    C1, DISCO's random degraded solution
- "rectified_flow_residue_heuristic": C2, the problem-specific degraded solution `residue(solver, g)`
"""
function build_method(method_name::String, scheduler::TimeSchedule, solver::ProblemSolver, residue_σ::Float32)
    if method_name == "rectified_flow_residue_random"
        return RectifiedFlow(scheduler; prediction=:x1, source=ResidueSource(solver; kind=:random, σ=residue_σ))
    elseif method_name == "rectified_flow_residue_heuristic"
        return RectifiedFlow(scheduler; prediction=:x1, source=ResidueSource(solver; kind=:heuristic, σ=residue_σ))
    else
        error("Unknown method_name: $method_name")
    end
end

"""
    execute_case(; ...)

Runs one grid cell: the first call for a `train_run_id` trains and saves the model, every later call loads it.
Seeding as in evaluate.jl (weight init by train_run_id, ATSS fit by run_id).
"""
function execute_case(; run_id::String, train_run_id::String, method_name::String, scheduler_name::String,
                       inference_steps::Int, generator::ERGraphGenerator, gnn_config::GNNConfig,
                       total_steps::Int, κ::Function, κ̇::Function, atss_p::Float32, residue_σ::Float32,
                       result_root::String, training_seed::Int, test_seed::Int, validation_seed::Int,
                       num_train::Int, num_test::Int, epochs::Int, batch_size::Int, learning_rate::Float64,
                       num_repeat_inference::Int, solver::ProblemSolver, evaluate::Bool=true)

    if scheduler_name == "atss"
        seed_everything!("atss:" * run_id)
    end
    scheduler = build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p)
    method = build_method(method_name, scheduler, solver, residue_σ)

    seed_everything!("init:" * train_run_id)
    model = DiffusionModel(AGNN(gnn_config), method) |> Flux.gpu

    return run_automated_pipeline(;
        run_id, train_run_id, model, generator, solver, num_train, num_test, epochs, batch_size,
        learning_rate, num_repeat_inference, evaluate, result_root, training_seed, test_seed, validation_seed,
    )
end

function main()
    result_root = "/home/e12023149/thesis/results"

    # ---- Must match evaluate.jl, so the runs are comparable with "rectified_flow" ----
    generator = ERGraphGenerator(min_nodes=50, max_nodes=100, p=0.15)
    solver_time_limit = 60.0
    gnn_config = GNNConfig(num_layers=12, in_dim=1, time_dim=32, hidden_dim=256, output_dim=1)
    total_steps = 1000
    κ  = t -> t^2.0f0                   # only used to fit ATSS (as for the other methods)
    κ̇  = t -> 2.0f0 * t^1.0f0
    atss_p = 0.5f0
    residue_σ = 1.0f0                   # noise on the degraded solution; DISCO: x₁ = X_d + ε, i.e. σ = 1

    num_train = 10000
    num_test = 1000
    epochs = 5                          # same training budget as all other methods
    batch_size = 32
    learning_rate = 0.001
    num_repeat_inference = 100

    seeds = Dict(
        "misp"   => (training = 5010001, test = 5010002, validation = 5010003),
        "maxcut" => (training = 5010004, test = 5010005, validation = 5010006),
    )

    # ---- The grid ----
    problems             = ["misp"]
    methods              = ["rectified_flow_residue_random", "rectified_flow_residue_heuristic"]
    schedulers           = ["linear", "cosine", "atss"]
    inference_steps_list = [1, 2, 5, 10, 20, 50]
    skip_existing_reports = true

    common = (; generator, gnn_config, total_steps, κ, κ̇, atss_p, residue_σ, result_root, num_train, num_test,
                epochs, batch_size, learning_rate, num_repeat_inference)

    for problem_name in problems
        solver = problem_name == "misp" ? MISPSolver(time_limit=solver_time_limit) :
                                           MaxCutSolver(time_limit=solver_time_limit)
        s = seeds[problem_name]
        split_seeds = (; training_seed = s.training, test_seed = s.test, validation_seed = s.validation)

        for method_name in methods
            train_id = build_train_run_id(problem_name, method_name, generator, gnn_config)

            # ---- Train (or load, if this train_id's model already exists); no evaluation ----
            @info "=== Training $train_id ==="
            try
                execute_case(; common..., split_seeds..., run_id = train_id, train_run_id = train_id, method_name,
                             scheduler_name = "linear", inference_steps = 2, solver, evaluate = false)
            catch e
                @error "Training $train_id FAILED — skipping its evaluations." exception=(e, catch_backtrace())
                continue
            end

            # ---- All evaluations for this model ----
            for scheduler_name in schedulers, inference_steps in inference_steps_list
                run_id = build_run_id(problem_name, method_name, generator, gnn_config, scheduler_name, inference_steps)
                if skip_existing_reports && isfile(joinpath(result_root, "reports", "report-$run_id.tex"))
                    @info "Skipping $run_id (report already exists)."
                    continue
                end

                @info "=== Evaluating $run_id ==="
                try
                    execute_case(; common..., split_seeds..., run_id, train_run_id = train_id, method_name,
                                 scheduler_name, inference_steps, solver)
                catch e
                    @error "Evaluation $run_id FAILED — continuing with the next cell." exception=(e, catch_backtrace())
                end
            end
        end
    end

    @info "Residue grid complete."
end

Base.invokelatest(main)
