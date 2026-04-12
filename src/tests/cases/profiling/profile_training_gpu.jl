# profile_training_gpu.jl
using Pkg
Pkg.activate(".")

module BenchmarkTestSetup
    using DotEnv
    using Thesis
    using CUDA
    using Flux
    using GraphNeuralNetworks
    using Statistics
    using Printf
    using Random

    env_path = joinpath(pkgdir(Thesis), "src", ".env")
    if isfile(env_path)
        DotEnv.load!(env_path)
    end

    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "instance_utils.jl"))
    include(joinpath(pkgdir(Thesis), "src", "utils", "data_utils.jl"))
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "test_utils.jl"))
    
    using .InstanceUtils
    using .DataUtils
    using .TestUtils
end

using Profile
using Thesis
using Flux
using CUDA # Explicitly required for CUDA.@profile
using .BenchmarkTestSetup.DataUtils: load_lazy_collection, split_lazy_dataset

function run_training_profile_gpu()
    instance_collection = "er_700_800"
    time_dim = 32
    feature_dim = 1
    batch_size = 4
    
    gnn_config = GNNConfig(
        num_layers=6,
        in_dim=feature_dim + time_dim,
        hidden_dim=128,
        output_dim=1
    )

    dataset, _ = load_lazy_collection(instance_collection)
    train_dataset, _ = split_lazy_dataset(dataset, 0.9)

    if length(train_dataset) > 0
        gnn = DifuscoGNN(gnn_config)
        diffusion_process = BernoulliDiffusion(100, "cosine")
        model = DiffusionModel(gnn, diffusion_process) |> Flux.gpu

        optim = Flux.setup(
            Flux.Optimisers.OptimiserChain(
                Flux.Optimisers.ClipNorm(1.0f0),
                Flux.Adam(1e-3)
            ),
            model.gnn
        ) 
        
        train_loader = Flux.DataLoader(train_dataset, batchsize=batch_size, shuffle=true, collate=true)
        
        # 1. WARM-UP: Run one epoch to trigger all Julia and CUDA compilation. 
        # Do not profile this, or your trace will be flooded with compiler overhead.
        @info "=== WARM-UP EPOCH (COMPILING) ==="
        Thesis.train_diffusion_model!(model, optim, train_loader, epochs=1)

        # 2. PROFILE: Run a second epoch wrapped in CUDA.@profile
        # This tells the Nsight Systems hook to actively record during this block.
        @info "=== PROFILED EPOCH ==="
        CUDA.@profile Thesis.train_diffusion_model!(model, optim, train_loader, epochs=1)
    end
end

# Execute the profiling sequence
run_training_profile_gpu()