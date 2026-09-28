using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CUDA
using cuDNN
using Flux
using Random
using DiFoG

include(joinpath(pkgdir(DiFoG), "test", "test_utils.jl"))

"""
    build_run_id(problem_name, method_name, generator, gnn_config, scheduler_name, inference_steps)

Builds an EVALUATION run_id, e.g. "misp_diffusion_er_50_100_12_256_atss_2". Identifies a report and a
test-instance dump. See `build_train_run_id` for the key that identifies the trained model itself.
"""
function build_run_id(problem_name::String, method_name::String, generator::ERGraphGenerator,
                       gnn_config::GNNConfig, scheduler_name::String, inference_steps::Int)
    return "$(problem_name)_$(method_name)_er_$(generator.min_nodes)_$(generator.max_nodes)_" *
           "$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_$(scheduler_name)_$(inference_steps)"
end

"""
    build_train_run_id(problem_name, method_name, generator, gnn_config)

Builds the TRAINING run_id — identifies the saved model file. Omits scheduler_name and inference_steps:
training only reads `.total_steps` of the schedule (uniform training time, no ρ(t)), so one trained
model is shared by every (scheduler, inference_steps) cell of a (problem, method).
"""
function build_train_run_id(problem_name::String, method_name::String, generator::ERGraphGenerator,
                             gnn_config::GNNConfig)
    return "$(problem_name)_$(method_name)_er_$(generator.min_nodes)_$(generator.max_nodes)_" *
           "$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_train"
end

"""
    atss_path(method_name)

The path ATSS is fitted on for a method: :diffusion, :flow_matching (also the Euler ablation) or :rectified_flow
(also Reflow).
"""
function atss_path(method_name::String)
    method_name == "diffusion" && return :diffusion
    startswith(method_name, "flow_matching") && return :flow_matching
    method_name in ("rectified_flow", "reflow") && return :rectified_flow
    error("Unknown method_name: $method_name")
end

"""
    build_scheduler(scheduler_name, method_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p)

Constructs the requested `TimeSchedule`. `generator`/`solver` are only used by "atss" (to fit the
schedule); Linear/Cosine ignore them. ATSS (Algorithm 1 of EFLOCO) is fitted on the path of the method:
the DFM mixture path (κ), the Bernoulli forward process of DDPM (κ = ᾱ), or the straight Gaussian path of
Rectified Flow / Reflow (κ(τ) = τ); see `atss_path`.
"""
function build_scheduler(scheduler_name::String, method_name::String, generator::GraphGenerator,
                          solver::ProblemSolver, total_steps::Int, inference_steps::Int, κ::Function,
                          κ̇::Function, atss_p::Float32)
    if scheduler_name == "linear"
        return LinearScheduler(total_steps, inference_steps)
    elseif scheduler_name == "cosine"
        return CosineScheduler(total_steps, inference_steps)
    elseif scheduler_name == "atss"
        return ATSSScheduler(generator, solver, total_steps, inference_steps, κ, κ̇; atss_p=atss_p,
                             path=atss_path(method_name))
    else
        error("Unknown scheduler_name: $scheduler_name")
    end
end

"""
    build_method(method_name, scheduler, κ, κ̇, λ_cons)

Constructs the requested `Diffusion` method. The method constructors convert the schedule themselves
(`BernoulliDiffusion` → `TimeSchedule{Int}` diffusion times, `DiscreteFlowMatching` → `TimeSchedule{Float32}`
progress; see the `TimeSchedule` time convention).

- "diffusion": Bernoulli diffusion (DIFUSCO)
- "flow_matching": DFM with the exact mixture-path step (sampling and consistency loss)
- "flow_matching_euler": DFM with the Euler step of EFLOCO (step-rule ablation). The step rule also
  enters the consistency loss, so this trains a separate model.
- "rectified_flow": continuous Rectified Flow (1-RF) on the {−1, +1} relaxation, endpoint parameterization
- "reflow": 2-RF, trained on the ODE coupling of the "rectified_flow" model (see `reflow_teacher`)
"""
function build_method(method_name::String, scheduler::TimeSchedule, κ::Function, κ̇::Function, λ_cons::Float32)
    if method_name == "diffusion"
        return BernoulliDiffusion(scheduler)
    elseif method_name == "flow_matching"
        return DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇; sampler=:exact)
    elseif method_name == "flow_matching_euler"
        return DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇; sampler=:euler)
    elseif method_name == "rectified_flow" || method_name == "reflow"
        return RectifiedFlow(scheduler; prediction=:x1)
    else
        error("Unknown method_name: $method_name")
    end
end

"""
    reflow_teacher(method_name)

Method whose trained model defines the Reflow coupling for `method_name`, or `nothing` for methods that
are trained on the data labels. "reflow" must therefore run after "rectified_flow" in the grid.
"""
reflow_teacher(method_name::String) = method_name == "reflow" ? "rectified_flow" : nothing

"""
    execute_case(; ...)

Runs one grid cell. `run_id` (report/instances) and `train_run_id` (model file) are passed separately
to `run_automated_pipeline`: the first call for a `train_run_id` trains and saves the model, every later
call loads it. Evaluation cells whose report already exists are skipped in `main` (resume). `evaluate=false` is used for the pure training call (R32: no evaluation of the
placeholder schedule).

Seeding (R31): the weight initialization depends only on `train_run_id`, the ATSS fit only on
`run_id`; training and evaluation randomness are seeded inside `run_automated_pipeline`.

`comparison_solver` (optional) labels the test instances a second time for comparison in the report
(e.g. Gurobi next to the LNS reference); `nothing` = no comparison.
"""
function execute_case(; run_id::String, train_run_id::String, method_name::String, scheduler_name::String,
                       inference_steps::Int, generator::ERGraphGenerator,
                       gnn_config::GNNConfig, total_steps::Int, κ::Function, κ̇::Function, atss_p::Float32,
                       λ_cons::Float32, result_root::String, training_seed::Int, test_seed::Int,
                       validation_seed::Int, num_train::Int, num_test::Int, epochs::Int, batch_size::Int,
                       learning_rate::Float64, num_repeat_inference::Int,
                       solver::ProblemSolver, evaluate::Bool=true,
                       reflow_from::Union{Nothing,String}=nothing, reflow_ode_steps::Int=100,
                       reflow_pairs_per_instance::Int=1,
                       comparison_solver::Union{Nothing,ProblemSolver}=nothing,
                       comparison_solvers::AbstractVector=ProblemSolver[])

    if scheduler_name == "atss"
        seed_everything!("atss:" * run_id)
    end
    scheduler = build_scheduler(scheduler_name, method_name, generator, solver, total_steps, inference_steps,
                                κ, κ̇, atss_p)
    method = build_method(method_name, scheduler, κ, κ̇, λ_cons)

    seed_everything!("init:" * train_run_id)
    gnn = AGNN(gnn_config)
    model = DiffusionModel(gnn, method) |> Flux.gpu

    return run_automated_pipeline(;
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
        evaluate = evaluate,
        result_root = result_root,
        training_seed = training_seed,
        test_seed = test_seed,
        validation_seed = validation_seed,
        reflow_from = reflow_from,
        reflow_ode_steps = reflow_ode_steps,
        reflow_pairs_per_instance = reflow_pairs_per_instance,
        comparison_solver = comparison_solver,
        comparison_solvers = comparison_solvers,
    )
end

function main()
    DiFoG.GRB_THREADS[] = 1                    # S12: Gurobi labels/comparisons with 1 thread, independent of the machine
    result_root = "/home/e12023149/thesis/results/results_50_100"

    # ---- Fixed across the entire grid ----
    generator = ERGraphGenerator(min_nodes=50, max_nodes=100, p=0.15)
    # Labels and reference: one solver per distribution, the one with the better solutions on average: Gurobi on
    # ER-50-100 (solves the instances exactly or nearly so), the LNS on the larger distributions. The other solver
    # labels the test instances as comparison. The LNS is LNS-rand of Bresich et al. with a single random repair per
    # iteration (r = 1) and an iteration limit instead of a time limit, so its labels do not depend on the machine;
    # 10^5 iterations are converged on all distributions (test/helpers/lns_calibration.jl).
    reference = :gurobi                        # :gurobi or :lns
    solver_time_limit = 10.0                   # Gurobi time limit (1 thread, conan)
    lns_iterations = 100_000                   # LNS iteration limit
    lns_repairs = 1                            # random repairs per LNS iteration (paper: 2000 MIS / 1000 MaxCut)
    # in_dim = node state width (1), time_dim = timestep embedding width (32); same weights as the old in_dim=33
    gnn_config = GNNConfig(num_layers=12, in_dim=1, time_dim=32, hidden_dim=256, output_dim=1)
    total_steps = 1000
    κ  = t -> t^2.0f0                          # shared by ATSS fitting AND DiscreteFlowMatching
    κ̇  = t -> 2.0f0 * t^1.0f0
    atss_p = 0.5f0
    λ_cons = 0.1f0
    reflow_ode_steps = 100                     # Euler steps generating the Reflow coupling
    reflow_pairs_per_instance = 1              # (x₀, Z₁) pairs per training instance, each with a fresh x₀

    num_train = 10000
    num_test = 1000
    epochs = 5
    batch_size = 32
    learning_rate = 0.001
    num_repeat_inference = 100

    # ---- Seeds: one triple per PROBLEM ----
    seeds = Dict(
        "misp"   => (training = 5010011, test = 5010012, validation = 5010013),
        "maxcut" => (training = 5010014, test = 5010015, validation = 5010016),
    )

    # ---- The grid ----
    problems             = ["maxcut", "misp"]
    # "reflow" must come after "rectified_flow" (it is trained on that model's ODE coupling);
    # add "flow_matching_euler" for the step-rule ablation
    methods              = ["diffusion", "flow_matching", "rectified_flow", "reflow"]
    schedulers           = ["linear", "cosine", "atss"]
    inference_steps_list = [1, 2, 5, 10, 20, 50]

    # H4: on a rerun, cells whose report already exists are skipped (training is skipped anyway when the
    # model file exists). Set to false to force re-evaluation.
    skip_existing_reports = true

    n_evals_per_model = length(schedulers) * length(inference_steps_list)
    n_models = length(problems) * length(methods)
    model_num = 0

    common = (; generator, gnn_config, total_steps, κ, κ̇, atss_p, λ_cons, result_root,
                num_train, num_test, epochs, batch_size, learning_rate, num_repeat_inference, reflow_ode_steps,
                reflow_pairs_per_instance)

    # One (problem, method) at a time: TRAIN once, then run all its evaluations.
    for problem_name in problems
        exact_solver = problem_name == "misp" ? MISPSolver(time_limit=solver_time_limit) :
                                                 MaxCutSolver(time_limit=solver_time_limit)
        lns_solver = LNSSolver(exact_solver; ttime=-1, titer=lns_iterations, n_iter_repair=lns_repairs)
        solver = reference === :lns ? lns_solver : exact_solver
        comparison_solver = reference === :lns ? exact_solver : lns_solver   # the other solver, on the test set
        comparison_solvers = ProblemSolver[]
        s = seeds[problem_name]
        split_seeds = (; training_seed = s.training, test_seed = s.test, validation_seed = s.validation)

        for method_name in methods
            model_num += 1
            train_id = build_train_run_id(problem_name, method_name, generator, gnn_config)
            teacher = reflow_teacher(method_name)
            reflow_from = isnothing(teacher) ? nothing : build_train_run_id(problem_name, teacher, generator, gnn_config)

            # ---- Train (or load, if this train_id's model already exists); no evaluation (R32) ----
            @info "=== Model $model_num/$n_models: training $train_id ==="
            try
                execute_case(; common..., split_seeds...,
                    run_id = train_id,
                    train_run_id = train_id,
                    method_name = method_name,
                    scheduler_name = "linear",       # placeholder: training only reads total_steps
                    inference_steps = 2,
                    solver = solver,
                    evaluate = false,
                    reflow_from = reflow_from,
                )
            catch e
                @error "Training $train_id FAILED — skipping its evaluations and continuing with the next model." exception=(e, catch_backtrace())
                continue
            end

            # ---- All evaluations for THIS model ----
            eval_num = 0
            for scheduler_name in schedulers, inference_steps in inference_steps_list
                eval_num += 1
                run_id = build_run_id(problem_name, method_name, generator, gnn_config, scheduler_name, inference_steps)

                if skip_existing_reports && isfile(joinpath(result_root, "reports", "report-$run_id.tex"))
                    @info "Skipping $run_id (report already exists)."
                    continue
                end

                @info "=== Model $model_num/$n_models, eval $eval_num/$n_evals_per_model: $run_id ==="

                try
                    execute_case(; common..., split_seeds...,
                        run_id = run_id,
                        train_run_id = train_id,
                        method_name = method_name,
                        scheduler_name = scheduler_name,
                        inference_steps = inference_steps,
                        solver = solver,
                        reflow_from = reflow_from,
                        comparison_solver = comparison_solver,
                        comparison_solvers = comparison_solvers,
                    )
                catch e
                    @error "Evaluation $run_id FAILED — logging and continuing with the next combination." exception=(e, catch_backtrace())
                end
            end

            @info "=== Model $model_num/$n_models ($train_id) complete: $eval_num/$n_evals_per_model evaluations attempted. ==="
        end
    end

    @info "Grid complete: $model_num/$n_models models, $n_evals_per_model evaluations each."
end

Base.invokelatest(main)
