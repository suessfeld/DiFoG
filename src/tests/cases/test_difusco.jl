using TestItems
using Thesis

@testitem "DIFUSCO TRAIN MODEL" setup=[DIFUSCOTestSetup] begin
    using Profile
    using Thesis
    using Flux
    
    # DIFUSCO CONFIG
    epochs = 5 # paper = 50
    learning_rate = 1e-3
    batch_size = 16 # paper = 32
    time_dim = 32
    feature_dim = 1
    gnn_config = GNNConfig(
        num_layers=6, # paper = 12
        in_dim=feature_dim + time_dim,
        hidden_dim=128, # paper = 256
        output_dim=1
    )
    
    generator = ERGraphGenerator(min_nodes=50, max_nodes=100, p=0.15)
    solver = MISPSolver(time_limit=0.1)
    train_dataset = RuntimeDataset(1000, generator, solver) 
    
    @info "Runtime dataset initialized"
    
    if length(train_dataset) > 0
        gnn = DifuscoGNN(gnn_config)
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
        loss_history = Thesis.train_diffusion_model!(model, optim, train_loader, epochs=epochs)
                
        # save model
        models_dir = joinpath(pkgdir(Thesis), ENV["MODELS_PATH"])
        mkpath(models_dir)
        model_path = joinpath(models_dir, "difusco_bernoulli_test.jld2")
        
        Thesis.save_model(model, model_path)
    end
end

@testitem "TEST DIFUSCO MODEL" setup=[DIFUSCOTestSetup] begin
    using Profile
    using Thesis
    using Flux
    using .DIFUSCOTestSetup.TestUtils: run_inference_and_save

    # same config as before! Note that THIS has to be the same as for the training!
    time_dim = 32
    feature_dim = 1
    gnn_config = GNNConfig(
        num_layers=12,
        in_dim=feature_dim + time_dim,
        hidden_dim=256,
        output_dim=1
    )

    # Setup Runtime Dataset for Testing
    config = (min_nodes=50, max_nodes=100, er_p=0.15)
    generator = ERGraphGenerator()
    solver = MISPSolver()
    
    test_dataset = RuntimeDataset(100, generator, solver, config)
    
    inferences_dir = joinpath(pkgdir(Thesis), "test_inferences", "difusco_instances_small")
    mkpath(inferences_dir)

    if length(test_dataset) > 0
        gnn = DifuscoGNN(gnn_config)
        diffusion_process = BernoulliDiffusion(100, "cosine")
        model = DiffusionModel(gnn, diffusion_process)

        # load model and move it to GPU
        models_dir = joinpath(pkgdir(Thesis), ENV["MODELS_PATH"])
        model_path = joinpath(models_dir, "difusco_bernoulli_test.jld2")
        Thesis.load_model!(model, model_path)
        model = model |> Flux.gpu

        @info "Running Inference and saving results..." 
        run_inference_and_save(model, test_dataset, inferences_dir, 4)
        @info "Inferences successfully saved to: $inferences_dir"
    end
end

@testmodule DIFUSCOTestSetup begin
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

    include(joinpath(pkgdir(Thesis), "src", "tests", "test_utils.jl"))
    
    using .TestUtils
end