# profile_run.jl
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
using .BenchmarkTestSetup.DataUtils: load_lazy_collection, split_lazy_dataset

function profile_training_cpu()
    instance_collection = "er_700_800"
    time_dim = 32
    feature_dim = 1
    batch_size = 4
    epochs = 1 # Keep it to 1 epoch for profiling
    
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
        
        # Run training
        Thesis.train_diffusion_model!(model, optim, train_loader, epochs=epochs)
    end
end
# for compiling
profile_training_cpu()

#actual profiling
@profview profile_training_cpu()