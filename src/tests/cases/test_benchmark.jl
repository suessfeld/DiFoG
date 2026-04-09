using TestItems
using Thesis

@testitem "TRAIN MODEL" setup=[BenchmarkTestSetup] begin
    using Profile
    using Thesis
    using Flux
    using .BenchmarkTestSetup.DataUtils: load_lazy_collection, split_lazy_dataset
    
    instance_collection="er_700_800"
    epochs = 2
    learning_rate = 1e-3
    batch_size = 8
    time_dim = 32
    feature_dim = 1
    gnn_config = GNNConfig(
        num_layers=12,
        in_dim=feature_dim + time_dim,
        hidden_dim=256,
        output_dim=1
    )
    
    dataset, _ = load_lazy_collection(instance_collection)
    
    total_instances = length(dataset)
    @info "LazyLoaded $total_instances instance paths total."
    
    train_dataset, test_dataset = split_lazy_dataset(dataset, 0.9)
    @info "Split: $(length(train_dataset)) for training, $(length(test_dataset)) for testing."
    
    if length(train_dataset) > 0
        gnn = AnisotropicGNN(gnn_config)
        diffusion_process = BernoulliDiffusion(100, "cosine")
        
        model = DiffusionModel(gnn, diffusion_process) |> Flux.gpu

        optim = Flux.setup(
            Flux.Optimisers.OptimiserChain(
                Flux.Optimisers.ClipNorm(1.0f0),   # clip global gradient norm to 1.0
                Flux.Adam(learning_rate)
            ),
            model.gnn
        ) 
        
        train_loader = Flux.DataLoader(train_dataset, batchsize=batch_size, shuffle=true, collate=true)
        batch_types = [typeof(first(b)) for b in Iterators.take(train_loader, 5)]
        @assert allequal(batch_types) "Graph types still vary — normalization incomplete"

        loss_history = Thesis.train_diffusion_model!(model, optim, train_loader, epochs=epochs)
                
        # save model
        models_dir = joinpath(pkgdir(Thesis), ENV["MODELS_PATH"])
        mkpath(models_dir)
        model_path = joinpath(models_dir, "$instance_collection.jld2")
        
        Thesis.save_model(model, model_path)
    end
end

@testitem "TEST MODEL" setup=[BenchmarkTestSetup] begin
    using Profile
    using Thesis
    using Flux
    using .BenchmarkTestSetup.DataUtils: load_lazy_collection, split_lazy_dataset
    using .BenchmarkTestSetup.TestUtils: run_inference_and_save

    instance_collection="er_700_800"
    time_dim = 32
    feature_dim = 1
    gnn_config = GNNConfig(
        num_layers=12,
        in_dim=feature_dim + time_dim,
        hidden_dim=256,
        output_dim=1
    )

    dataset, inferences_dir = load_lazy_collection(instance_collection)
    
    _, test_dataset = split_lazy_dataset(dataset, 0.9)

    if length(test_dataset) > 0
        gnn = AnisotropicGNN(gnn_config)
        diffusion_process = BernoulliDiffusion(100, "cosine")
        model = DiffusionModel(gnn, diffusion_process)

        # load model and move it to GPU
        models_dir = joinpath(pkgdir(Thesis), ENV["MODELS_PATH"])
        model_path = joinpath(models_dir, "$instance_collection.jld2")
        Thesis.load_model!(model, model_path)
        model = model |> Flux.gpu

        @info "Running Inference and saving results..." 
        run_inference_and_save(model, test_dataset, inferences_dir, 2)
        @info "Inferences successfully saved to: $inferences_dir"
    end
end

@testmodule BenchmarkTestSetup begin
    using DotEnv
    using Thesis
    using CUDA
    using Flux
    using GraphNeuralNetworks
    using Statistics
    using Graphs
    using Printf
    using Random

    env_path = joinpath(pkgdir(Thesis), "src", ".env")
    if isfile(env_path)
        DotEnv.load!(env_path)
    else
        @warn "No .env file found at $env_path"
    end

    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "instance_utils.jl"))
    include(joinpath(pkgdir(Thesis), "src", "utils", "data_utils.jl"))
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "test_utils.jl"))
    
    using .InstanceUtils
    using .DataUtils
    using .TestUtils
end