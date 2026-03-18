# src/tests/cases/test_gaussian.jl
using TestItems
using Thesis

@testitem "Train Gaussian Diffusion on MISP" setup=[GaussianTestSetup] begin
    using Flux
    using Thesis
    using .GaussianTestSetup.TestUtils: load_collection, split_dataset, train_diffusion_model!, run_inference_and_save
    
    instance_collection="misp_medium_instances"
    epochs = 5
    learning_rate = 1e-3

    time_dim = 32
    feature_dim = 1
    gnn_config = GNNConfig(
        num_layers=4,
        in_dim=feature_dim + time_dim,
        hidden_dim=64,
        output_dim=1
    )
    
    all_pairs, inferences_dir = load_collection(instance_collection)
    
    total_instances = length(all_pairs)
    @info "Loaded $total_instances instances total."
    
    training_pairs, test_pairs = split_dataset(all_pairs, 0.9)
    @info "Split: $(length(training_pairs)) for training, $(length(test_pairs)) for testing."
    
    if !isempty(training_pairs)
        gnn = AnisotropicGNN(gnn_config)
        diffusion_process = GaussianDiffusion(100, "cosine")
        model = DiffusionModel(gnn, diffusion_process)
        
        optim = Flux.setup(Flux.Adam(learning_rate), model) # TODO Adam ok?
        
        @info "Starting Training..."
        loss_history = train_diffusion_model!(model, optim, training_pairs, epochs=epochs)
        
        @test loss_history[end] < loss_history[1]
        
        @info "Running Inference and saving results..."  # TODO Probably add timing
        run_inference_and_save(model, test_pairs, inferences_dir)
        @info "Inferences successfully saved to: $inferences_dir"
    end
end

@testitem "Train Gaussian Diffusion on MAXDEG" setup=[GaussianTestSetup] begin
    using Flux
    using Thesis
    using .GaussianTestSetup.TestUtils: load_collection, split_dataset, train_diffusion_model!, run_inference_and_save
    
    instance_collection="maxdeg_large_instances"
    epochs = 5
    learning_rate = 1e-3

    time_dim = 32
    feature_dim = 1
    gnn_config = GNNConfig(
        num_layers=4,
        in_dim=feature_dim + time_dim,
        hidden_dim=64,
        output_dim=1
    )
    
    all_pairs, inferences_dir = load_collection(instance_collection)
    
    total_instances = length(all_pairs)
    @info "Loaded $total_instances instances total."
    
    training_pairs, test_pairs = split_dataset(all_pairs, 0.9)
    @info "Split: $(length(training_pairs)) for training, $(length(test_pairs)) for testing."
    
    if !isempty(training_pairs)
        gnn = AnisotropicGNN(gnn_config)
        diffusion_process = GaussianDiffusion(100, "cosine")
        model = DiffusionModel(gnn, diffusion_process)
        
        optim = Flux.setup(Flux.Adam(learning_rate), model)
        
        @info "Starting Training..."
        loss_history = train_diffusion_model!(model, optim, training_pairs, epochs=epochs)
        
        @test loss_history[end] < loss_history[1]
        
        @info "Running Inference and saving results..."
        run_inference_and_save(model, test_pairs, inferences_dir)
        @info "Inferences successfully saved to: $inferences_dir"
    end
end

@testmodule GaussianTestSetup begin
    using DotEnv
    using Thesis
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
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "test_utils.jl"))
    
    using .InstanceUtils
    using .TestUtils
end