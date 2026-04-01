# src/tests/cases/test_bernoulli_profiler.jl

using Profile
using DotEnv
using cuDNN
using CUDA
using Flux
using GraphNeuralNetworks
using Statistics
using Printf
using Random
using Thesis

env_path = joinpath(pkgdir(Thesis), "src", ".env")
if isfile(env_path)
    DotEnv.load!(env_path)
else
    @warn "No .env file found at $env_path"
end

module ProfilerSetup
    using Thesis
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "instance_utils.jl"))
    include(joinpath(pkgdir(Thesis), "src", "tests", "utils", "test_utils.jl"))
end

using .ProfilerSetup.TestUtils: load_collection, split_dataset, train_diffusion_model!, run_inference_and_save

function run_profiler()
    instance_collection = "misp_medium_instances"
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
        diffusion_process = BernoulliDiffusion(100, "cosine")

        # Move the model to the GPU
        model = DiffusionModel(gnn, diffusion_process) |> Flux.gpu
        optim = Flux.setup(Flux.Adam(learning_rate), model)

        training_pairs_gpu = [(g |> Flux.gpu, name) for (g, name) in training_pairs]
        test_pairs_gpu = [(g |> Flux.gpu, name) for (g, name) in test_pairs]

        # warmup for correct profiling
        train_diffusion_model!(model, optim, training_pairs_gpu[1:min(2, end)], epochs=1)

        loss_history = @profview train_diffusion_model!(model, optim, training_pairs_gpu, epochs=epochs)

        if loss_history[end] < loss_history[1]
            @info "Success: Loss decreased from $(loss_history[1]) to $(loss_history[end])"
        else
            @warn "Loss did not decrease."
        end

        @info "Running Inference and saving results..."
        run_inference_and_save(model, test_pairs_gpu, inferences_dir)
        @info "Inferences successfully saved to: $inferences_dir"
    end
end

run_profiler()