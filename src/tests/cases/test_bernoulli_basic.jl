# src/tests/cases/test_bernoulli_CUDA.jl
using TestItems
using Thesis

@testitem "Train Gaussian Diffusion on MISP" setup=[BernoulliTestSetupDIFUSCO] begin
    using Profile
    using CUDA
    using Flux
    using Thesis
    using GraphNeuralNetworks
    using .BernoulliTestSetupDIFUSCO.TestUtils: load_collection, split_dataset, train_diffusion_model!, run_inference_and_save
    
    instance_collection="difusco_instances"
    epochs = 5
    learning_rate = 1e-3
    batch_size = 64 # Configure your batch size to fit your VRAM

    time_dim = 32
    feature_dim = 1
    gnn_config = GNNConfig(
        num_layers=12,
        in_dim=feature_dim + time_dim,
        hidden_dim=256,
        output_dim=1
    )
    
    all_pairs, inferences_dir = load_collection(instance_collection)
    
    total_instances = length(all_pairs)
    @info "Loaded $total_instances instances total."
    
    training_pairs, test_pairs = split_dataset(all_pairs, 0.9)
    @info "Split: $(length(training_pairs)) for training, $(length(test_pairs)) for testing."
    
    if !isempty(training_pairs)
        gnn = AnisotropicGNN(gnn_config)
        diffusion_process = BernoulliDiffusion(100, "cosine")
        
        # move the model to the GPU
        model = DiffusionModel(gnn, diffusion_process) |> Flux.gpu

        optim = Flux.setup(Flux.Adam(learning_rate), model) 
        
        # 1. Create a DataLoader for lazy batching on the CPU
        # By passing a vector of tuples (GNNGraph, String), MLUtils/Flux automatically
        # batches the graphs into disjoint graphs and collates the strings.
        train_loader = Flux.DataLoader(training_pairs, batchsize=batch_size, shuffle=true)
        
        @info "Starting Training..."
        # 2. Pass the DataLoader instead of the full array
        loss_history = train_diffusion_model!(model, optim, train_loader, epochs=epochs)
        
        @test loss_history[end] < loss_history[1]
        
        @info "Running Inference and saving results..." 
        # 3. Pass CPU test pairs (just-in-time GPU transfer happens inside)
        run_inference_and_save(model, test_pairs, inferences_dir)
        @info "Inferences successfully saved to: $inferences_dir"
    end
end

@testmodule BernoulliTestSetupDIFUSCO begin
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
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "test_utils.jl"))
    
    using .InstanceUtils
    using .TestUtils
end