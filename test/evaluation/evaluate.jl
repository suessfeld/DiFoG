using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CUDA
using cuDNN
using Flux
using Random
using Thesis

include(joinpath(pkgdir(Thesis), "test", "test_utils.jl"))

"""
    build_run_id(problem_name, method_name, generator, gnn_config, scheduler_name, inference_steps)

Builds an EVALUATION run_id, e.g. "misp_diffusion_er_50_100_12_256_atss_2". Identifies a report,
loss plot (if this cell happens to be the one that trains), and test-instance dump. See
`build_train_run_id` for the (coarser) key that identifies the trained model itself.
"""
function build_run_id(problem_name::String, method_name::String, generator::ERGraphGenerator,
                       gnn_config::GNNConfig, scheduler_name::String, inference_steps::Int)
    return "$(problem_name)_$(method_name)_er_$(generator.min_nodes)_$(generator.max_nodes)_" *
           "$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_$(scheduler_name)_$(inference_steps)"
end

"""
    build_train_run_id(problem_name, method_name, generator, gnn_config)

Builds the TRAINING run_id — identifies the saved model file. Deliberately omits scheduler_name and
inference_steps: `corrupt_batch`/`compute_loss_and_gradients` only ever read `.total_steps` off a
scheduler (see project notes), so two schedulers differing only in type/inference_steps are
training-equivalent. One trained model is shared across every (scheduler, inference_steps)
combination for a given (problem, method) — 4 trainings total instead of 60.
"""
function build_train_run_id(problem_name::String, method_name::String, generator::ERGraphGenerator,
                             gnn_config::GNNConfig)
    return "$(problem_name)_$(method_name)_er_$(generator.min_nodes)_$(generator.max_nodes)_" *
           "$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_train"
end

"""
    build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p)

Constructs the requested scheduler type. `generator`/`solver` are only used by "atss" (needed to fit
the schedule); Linear/Cosine ignore them but accept them for a uniform call signature.
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
    build_method(method_name, scheduler, κ, κ̇, λ_cons)

Constructs the requested `Diffusion` method. Julia's automatic struct-field `convert` adapts
`scheduler` to whatever `Scheduler{T}` each method's struct field declares (`Scheduler{Int}` for
BernoulliDiffusion, `Scheduler{Float32}` for DiscreteFlowMatching) — see the B6 fix in atss.jl for
the ATSS-specific Int-conversion path this relies on.
"""
function build_method(method_name::String, scheduler, κ::Function, κ̇::Function, λ_cons::Float32)
    if method_name == "diffusion"
        return BernoulliDiffusion(scheduler)
    elseif method_name == "flow_matching"
        return DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇)
    else
        error("Unknown method_name: $method_name")
    end
end

"""
    execute_case(; ...)

Runs one grid-search cell. `run_id` (eval identity: report/loss-plot/instances) and `train_run_id`
(model identity, see `build_train_run_id`) are passed separately to `run_automated_pipeline` — the
first real call for a given `train_run_id` performs actual training and saves the model; every
subsequent call with the same `train_run_id` loads those weights and only re-runs evaluation with
its own scheduler. `run_automated_pipeline`'s own "skip training if model already exists" check means
a crashed/interrupted grid search resumes cleanly on rerun — already-completed cells (train OR eval)
are skipped, not redone.
"""
function execute_case(; run_id::String, train_run_id::String, method_name::String, scheduler_name::String,
                       inference_steps::Int, generator::ERGraphGenerator, solver_time_limit::Float64,
                       gnn_config::GNNConfig, total_steps::Int, κ::Function, κ̇::Function, atss_p::Float32,
                       λ_cons::Float32, result_root::String, training_seed::Int, test_seed::Int,
                       validation_seed::Int, num_train::Int, num_test::Int, epochs::Int, batch_size::Int,
                       learning_rate::Float64, num_repeat_inference::Int,
                       solver::ProblemSolver, cached_scheduler=nothing)

    scheduler = isnothing(cached_scheduler) ?
        build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p) :
        cached_scheduler

    method = build_method(method_name, scheduler, κ, κ̇, λ_cons)

    gnn = AGNN(gnn_config)
    model = DiffusionModel(gnn, method) |> Flux.gpu

    run_automated_pipeline(;
        run_id = run_id,
        train_run_id = train_run_id,
        model = model,
        generator = generator,
        solver = solver,
        num_train = num_train,
        num_test = num_test,
        epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        num_repeat_inference = num_repeat_inference,
        result_root = result_root,
        training_seed = training_seed,
        test_seed = test_seed,
        validation_seed = validation_seed,
    )
end

function main()
    result_root = "/home/e12023149/thesis/results"

    # ---- Fixed across the entire grid ----
    generator = ERGraphGenerator(min_nodes=300, max_nodes=400, p=0.15)
    solver_time_limit = 60.0
    gnn_config = GNNConfig(num_layers=12, in_dim=33, hidden_dim=256, output_dim=1)
    total_steps = 1000
    κ  = t -> t^2.0f0                          # shared by ATSS fitting AND DiscreteFlowMatching's own equations
    κ̇  = t -> 2.0f0 * t^1.0f0
    atss_p = 0.5f0
    λ_cons = 0.1f0

    num_train = 10000
    num_test = 1000
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    num_repeat_inference = 100

    Random.seed!(1)
    CUDA.seed!(1)

    seeds = Dict(
        "misp"   => (training = 30040001, test = 30040002, validation = 30040003),
        "maxcut" => (training = 30040004, test = 30040005, validation = 3004006),
    )

    # grid
    problems              = ["maxcut", "misp"]
    methods                = ["diffusion", "flow_matching"]
    schedulers              = ["linear", "cosine", "atss"]
    inference_steps_list    = [2, 5, 10, 20, 50]

    n_evals_per_model = length(schedulers) * length(inference_steps_list)
    n_models = length(problems) * length(methods)
    model_num = 0

    for problem_name in problems
        solver = problem_name == "misp" ? MISPSolver(time_limit=solver_time_limit) :
                                           MaxCutSolver(time_limit=solver_time_limit)
        s = seeds[problem_name]

        for method_name in methods
            model_num += 1
            train_id = build_train_run_id(problem_name, method_name, generator, gnn_config)

            # ---- Train (or load, if this train_id's model already exists) ----
            @info "=== Model $model_num/$n_models: training $train_id ==="
            placeholder_scheduler = LinearScheduler(total_steps, 2)   # arbitrary — never read beyond .total_steps during training

            try
                execute_case(;
                    run_id = train_id,
                    train_run_id = train_id,
                    method_name = method_name,
                    scheduler_name = "linear",
                    inference_steps = 2,
                    generator = generator,
                    solver_time_limit = solver_time_limit,
                    gnn_config = gnn_config,
                    total_steps = total_steps,
                    κ = κ, κ̇ = κ̇,
                    atss_p = atss_p,
                    λ_cons = λ_cons,
                    result_root = result_root,
                    training_seed = s.training,
                    test_seed = s.test,
                    validation_seed = s.validation,
                    num_train = num_train,
                    num_test = num_test,
                    epochs = epochs,
                    batch_size = batch_size,
                    learning_rate = learning_rate,
                    num_repeat_inference = num_repeat_inference,
                    solver = solver,
                    cached_scheduler = placeholder_scheduler,
                )
            catch e
                @error "Training $train_id FAILED — skipping its evaluations and continuing with the next model." exception=(e, catch_backtrace())
                continue   # no point evaluating a model that never trained
            end

            eval_num = 0
            for scheduler_name in schedulers
                shared_scheduler = scheduler_name == "atss" ?
                    build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps_list[1], κ, κ̇, atss_p) :
                    nothing

                for inference_steps in inference_steps_list
                    eval_num += 1
                    scheduler_for_cell = scheduler_name == "atss" ?
                        build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p) :
                        nothing   # linear/cosine built fresh inside execute_case — cheap, no caching needed

                    run_id = build_run_id(problem_name, method_name, generator, gnn_config, scheduler_name, inference_steps)
                    @info "=== Model $model_num/$n_models, eval $eval_num/$n_evals_per_model: $run_id ==="

                    try
                        execute_case(;
                            run_id = run_id,
                            train_run_id = train_id,
                            method_name = method_name,
                            scheduler_name = scheduler_name,
                            inference_steps = inference_steps,
                            generator = generator,
                            solver_time_limit = solver_time_limit,
                            gnn_config = gnn_config,
                            total_steps = total_steps,
                            κ = κ, κ̇ = κ̇,
                            atss_p = atss_p,
                            λ_cons = λ_cons,
                            result_root = result_root,
                            training_seed = s.training,
                            test_seed = s.test,
                            validation_seed = s.validation,
                            num_train = num_train,
                            num_test = num_test,
                            epochs = epochs,
                            batch_size = batch_size,
                            learning_rate = learning_rate,
                            num_repeat_inference = num_repeat_inference,
                            solver = solver,
                            cached_scheduler = scheduler_for_cell,
                        )
                    catch e
                        @error "Evaluation $run_id FAILED — logging and continuing with the next combination." exception=(e, catch_backtrace())
                    end
                end
            end

            @info "=== Model $model_num/$n_models ($train_id) complete: $eval_num/$n_evals_per_model evaluations attempted. ==="
        end
    end

    @info "Grid complete: $model_num/$n_models models trained, $n_evals_per_model evaluations each."
end

Base.invokelatest(main)