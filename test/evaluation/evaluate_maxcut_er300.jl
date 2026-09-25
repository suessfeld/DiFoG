using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CUDA
using cuDNN
using Flux
using Random
using DiFoG

include(joinpath(pkgdir(DiFoG), "test", "test_utils.jl"))

# MaxCut on the ER-300 benchmark setting of Bresich et al., "Determining Destroy Sets in Large Neighborhood
# Search by Generative Flow Networks" (PPSN 2026): Erdős-Rényi graphs (Graphs.jl `erdos_renyi`) with n = 300
# and edge probability p = 0.1, 100 test instances. Labels and the reference are Gurobi solutions with the
# paper's ILP setting (3 s, single-threaded), so the reference mean should be close to the paper's ILP value.
#
# Reported values of the paper on its own ER-300 instances (mean objective ± σ, 100 instances), for comparison:
#   ILP (3 s)          2470.87 ± 41.20
#   ILP-30 (30 s)      2470.87 ± 41.20
#   LNS-rand (3 s)     2787.07 ± 36.43
#   GFN-LNS-TB (3 s)   2799.79 ± 36.97
# Our instances follow the same distribution but are different graphs: compare means ± σ, not per instance.

"""
    build_run_id(method_name, gnn_config, scheduler_name, inference_steps)

Builds an EVALUATION run_id, e.g. "maxcut_reflow_er300_p010_12_256_linear_1".
"""
build_run_id(method_name::String, gnn_config::GNNConfig, scheduler_name::String, inference_steps::Int) =
    "maxcut_$(method_name)_er300_p010_$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_$(scheduler_name)_$(inference_steps)"

"""
    build_train_run_id(method_name, gnn_config)

Builds the TRAINING run_id — identifies the saved model file, shared by all (scheduler, inference_steps) cells.
"""
build_train_run_id(method_name::String, gnn_config::GNNConfig) =
    "maxcut_$(method_name)_er300_p010_$(gnn_config.num_layers)_$(gnn_config.hidden_dim)_train"

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
    build_method(method_name, scheduler, κ, κ̇, λ_cons)

- "diffusion": Bernoulli diffusion (DIFUSCO)
- "flow_matching": DFM with consistency loss and the reference Euler sampler (EFLOCO)
- "rectified_flow": continuous Rectified Flow (1-RF), endpoint parameterization
- "reflow": 2-RF, trained on the ODE coupling of the "rectified_flow" model (see `reflow_teacher`)
"""
function build_method(method_name::String, scheduler::TimeSchedule, κ::Function, κ̇::Function, λ_cons::Float32)
    if method_name == "diffusion"
        return BernoulliDiffusion(scheduler)
    elseif method_name == "flow_matching"
        return DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇; sampler=:euler)
    elseif method_name == "rectified_flow" || method_name == "reflow"
        return RectifiedFlow(scheduler; prediction=:x1)
    else
        error("Unknown method_name: $method_name")
    end
end

"""
    reflow_teacher(method_name)

Method whose trained model defines the Reflow coupling, or `nothing`. "reflow" must run after "rectified_flow".
"""
reflow_teacher(method_name::String) = method_name == "reflow" ? "rectified_flow" : nothing

"""
    uses_scheduler(method_name, scheduler_name)

ATSS is fitted on the discrete κ mixture path, so it is only evaluated for "flow_matching".
"""
uses_scheduler(method_name::String, scheduler_name::String) =
    scheduler_name != "atss" || method_name == "flow_matching"

"""
    execute_case(; ...)

Runs one grid cell: the first call for a `train_run_id` trains and saves the model, every later call loads it.
"""
function execute_case(; run_id::String, train_run_id::String, method_name::String, scheduler_name::String,
                       inference_steps::Int, generator::ERGraphGenerator, gnn_config::GNNConfig,
                       total_steps::Int, κ::Function, κ̇::Function, atss_p::Float32, λ_cons::Float32,
                       result_root::String, training_seed::Int, test_seed::Int, validation_seed::Int,
                       num_train::Int, num_test::Int, epochs::Int, batch_size::Int, learning_rate::Float64,
                       num_repeat_inference::Int, solver::ProblemSolver, evaluate::Bool=true,
                       reflow_from::Union{Nothing,String}=nothing, reflow_ode_steps::Int=100)

    if scheduler_name == "atss"
        seed_everything!("atss:" * run_id)
    end
    scheduler = build_scheduler(scheduler_name, generator, solver, total_steps, inference_steps, κ, κ̇, atss_p)
    method = build_method(method_name, scheduler, κ, κ̇, λ_cons)

    seed_everything!("init:" * train_run_id)
    model = DiffusionModel(AGNN(gnn_config), method) |> Flux.gpu

    return run_automated_pipeline(;
        run_id, train_run_id, model, generator, solver, num_train, num_test, epochs, batch_size,
        learning_rate, num_repeat_inference, evaluate, result_root, training_seed, test_seed,
        validation_seed, reflow_from, reflow_ode_steps,
    )
end

function main()
    result_root = "/home/e12023149/thesis/results"

    # ---- Instances: ER-300 benchmark setting of Bresich et al. ----
    generator = ERGraphGenerator(min_nodes=300, max_nodes=300, p=0.1)

    # ---- Labels / reference: the paper's ILP setting (Gurobi, 3 s, single-threaded) ----
    solver = MaxCutSolver(time_limit=3.0)
    DiFoG.GRB_THREADS[] = 1

    # New seeds: seed files are keyed by the seed only, so they must not collide with other generators.
    training_seed, test_seed, validation_seed = 30300001, 30300002, 30300003

    gnn_config = GNNConfig(num_layers=12, in_dim=1, time_dim=32, hidden_dim=256, output_dim=1)
    total_steps = 1000
    κ  = t -> t^2.0f0
    κ̇  = t -> 2.0f0 * t^1.0f0
    atss_p = 0.5f0
    λ_cons = 0.1f0
    reflow_ode_steps = 100

    num_train = 10000
    num_test = 100                 # as in the paper
    epochs = 50
    batch_size = 16                # ER-300 with p = 0.1 has ~9000 directed edges per graph; 32 may exceed 24 GB
    learning_rate = 0.001
    num_repeat_inference = 100

    # ---- The grid ----
    methods              = ["diffusion", "flow_matching", "rectified_flow", "reflow"]   # reflow after rectified_flow
    schedulers           = ["linear", "cosine", "atss"]
    inference_steps_list = [1, 2, 5, 10, 20, 50]
    skip_existing_reports = true

    common = (; generator, gnn_config, total_steps, κ, κ̇, atss_p, λ_cons, result_root, training_seed,
                test_seed, validation_seed, num_train, num_test, epochs, batch_size, learning_rate,
                num_repeat_inference, solver, reflow_ode_steps)

    for (model_num, method_name) in enumerate(methods)
        train_id = build_train_run_id(method_name, gnn_config)
        teacher = reflow_teacher(method_name)
        reflow_from = isnothing(teacher) ? nothing : build_train_run_id(teacher, gnn_config)

        @info "=== Model $model_num/$(length(methods)): training $train_id ==="
        try
            execute_case(; common..., run_id = train_id, train_run_id = train_id, method_name,
                         scheduler_name = "linear", inference_steps = 2, evaluate = false, reflow_from)
        catch e
            @error "Training $train_id FAILED — skipping its evaluations and continuing with the next model." exception=(e, catch_backtrace())
            continue
        end

        for scheduler_name in schedulers, inference_steps in inference_steps_list
            uses_scheduler(method_name, scheduler_name) || continue
            run_id = build_run_id(method_name, gnn_config, scheduler_name, inference_steps)
            if skip_existing_reports && isfile(joinpath(result_root, "reports", "report-$run_id.tex"))
                @info "Skipping $run_id (report already exists)."
                continue
            end

            @info "=== Model $model_num/$(length(methods)): evaluating $run_id ==="
            try
                execute_case(; common..., run_id, train_run_id = train_id, method_name, scheduler_name,
                             inference_steps, reflow_from)
            catch e
                @error "Evaluation $run_id FAILED — logging and continuing with the next combination." exception=(e, catch_backtrace())
            end
        end
    end

    @info "Grid complete."
end

Base.invokelatest(main)
