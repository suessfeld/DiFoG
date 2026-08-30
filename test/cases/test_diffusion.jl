using TestItems
using Thesis

@testmodule TestDiffusion begin
    using Test
    using Thesis
    using GraphNeuralNetworks
    using Flux
end

@testitem "Test BernoulliDiffusion Corruption Step Statistical Distribution" setup=[TestDiffusion] begin
    using Flux
    using Statistics

    total_steps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, total_steps, inference_steps)
    diffusion = BernoulliDiffusion(scheduler)

    # 1. Setup a massive tensor to allow the Law of Large Numbers to converge
    N = 5000
    # Initialize completely with 1s so we can easily track the keep vs flip ratio
    x_0 = ones(Float32, N, N) |> Flux.gpu
    
    # Test at a specific timestep (e.g., halfway through the schedule)
    t_val = 50
    t = fill(Int32(t_val), N) |> Flux.gpu

    q_sampled = Thesis.q(diffusion, x_0, t) |> Flux.cpu

    # ---------------------------------------------------------
    # Theoretical Ground Truth Transition Probability[cite: 3]
    # ---------------------------------------------------------
    βs = Float32.(range(1f-4, 2f-2, length=total_steps))
    αs = cumprod(1.0f0 .- 2.0f0 .* βs)
    ᾱₜ = αs[t_val]
    
    # The theoretical probability of a bit keeping its original state[cite: 3]
    expected_prob_keep = 0.5f0 * (1.0f0 + ᾱₜ)

    # ---------------------------------------------------------
    # Empirical Validation
    # ---------------------------------------------------------
    # Because x_0 was entirely 1s, the mean of q_sampled represents the exact empirical ratio of kept bits
    empirical_prob_keep = mean(q_sampled)

    @info "Theoretical prob_keep:" expected_prob_keep
    @info "Empirical prob_keep:" empirical_prob_keep

    @test size(q_sampled) == (N, N)
    @test eltype(q_sampled) == Float32
    
    # Ensure it only produces valid categorical {0, 1} states[cite: 3]
    @test all(x -> x == 0.0f0 || x == 1.0f0, q_sampled)
    
    # Assert the sampled distribution matches the paper's theoretical equation within a 0.5% tolerance
    @test isapprox(empirical_prob_keep, expected_prob_keep, atol=5e-3)
end

@testitem "Test BernoulliDiffusion Posterior Probability exact values" setup=[TestDiffusion] begin
    using Flux
    using Statistics

    total_steps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, total_steps, inference_steps)
    diffusion = BernoulliDiffusion(scheduler)

    # Move mock inputs to GPU
    x_t = Float32[1.0 0.0; 0.0 1.0] |> Flux.gpu
    x_0_pred = Float32[0.9 0.1; 0.2 0.8] |> Flux.gpu
    t_idx = 50
    s_idx = 40

    p_post = Thesis.posterior_prob(diffusion, x_t, x_0_pred, t_idx, s_idx) |> Flux.cpu

    βs = Float32.(range(1f-4, 2f-2, length=total_steps))
    αs = cumprod(1.0f0 .- 2.0f0 .* βs)
    
    # Extract cumulative scaling factors \bar{\alpha}_t and \bar{\alpha}_s[cite: 3]
    α_bar_t = αs[t_idx]
    α_bar_s = αs[s_idx]
    α_t_s = α_bar_t / α_bar_s
    
    # q(x_t = 1 | x_{t-1} = 1) 
    p_t_s = 0.5f0 * (1.0f0 + α_t_s)
    
    # Calculate expected exact values for each matrix element independently
    expected = zeros(Float32, 2, 2)
    x_t_cpu = x_t |> Flux.cpu
    x_0_pred_cpu = x_0_pred |> Flux.cpu
    
    for i in 1:2
        for j in 1:2
            xt_val = x_t_cpu[i, j]
            x0_pred_val = x_0_pred_cpu[i, j]
            
            # q(x_t | x_{t-1} = 1)
            term1 = xt_val == 1.0f0 ? p_t_s : (1.0f0 - p_t_s)
            
            # Eq 5 evaluated for x_0 = 1: q(x_{t-1}=1 | x_t, x_0=1)[cite: 3]
            denom_1 = xt_val == 1.0f0 ? 0.5f0 * (1.0f0 + α_bar_t) : 0.5f0 * (1.0f0 - α_bar_t)
            term2 = (term1 * 0.5f0 * (1.0f0 + α_bar_s)) / denom_1
            
            # Eq 5 evaluated for x_0 = 0: q(x_{t-1}=1 | x_t, x_0=0)[cite: 3]
            denom_0 = xt_val == 1.0f0 ? 0.5f0 * (1.0f0 - α_bar_t) : 0.5f0 * (1.0f0 + α_bar_t)
            term3 = (term1 * 0.5f0 * (1.0f0 - α_bar_s)) / denom_0
            
            # Eq 6: p_θ(x_{t-1}=1 | x_t)[cite: 3]
            expected[i, j] = clamp(x0_pred_val * term2 + (1.0f0 - x0_pred_val) * term3, 0.0f0, 1.0f0)
        end
    end

    @test size(p_post) == size(x_t |> Flux.cpu)
    @test all(x -> 0.0f0 <= x <= 1.0f0, p_post)
    @test p_post ≈ expected atol=1e-6
end

@testitem "Test BernoulliDiffusion Posterior Probability boundary case s=0" setup=[TestDiffusion] begin
    using Flux

    total_steps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, total_steps, inference_steps)
    diffusion = BernoulliDiffusion(scheduler)

    # Move mock inputs to GPU. (otherwise not supported!)
    x_t = Float32[1.0 0.0] |> Flux.gpu
    x_0_pred = Float32[0.99 0.01] |> Flux.gpu
    t_idx = 10
    s_idx = 0

    p_post = Thesis.posterior_prob(diffusion, x_t, x_0_pred, t_idx, s_idx) |> Flux.cpu

    @info p_post
    @test size(p_post) == size(x_t |> Flux.cpu)
    @test all(x -> 0.0f0 <= x <= 1.0f0, p_post)
end