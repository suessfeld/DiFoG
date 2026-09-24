using TestItems
using Thesis

@testmodule TestDiffusion begin
    using Test
    using Thesis
    using GraphNeuralNetworks
    using Flux

    # Reference β schedule (DIFUSCO linear 1e-4 → 2e-2 for T = 1000, endpoints scaled by 1000/T),
    # written out independently of Thesis.get_betas on purpose.
    reference_alphas(T) = cumprod(1.0f0 .- 2.0f0 .* Float32.(range(1f-4 * 1000f0 / T, 2f-2 * 1000f0 / T, length=T)))
    export reference_alphas
end

@testitem "Test BernoulliDiffusion Corruption Step Statistical Distribution" setup=[TestDiffusion] begin
    using Flux
    using Statistics

    total_steps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, total_steps, inference_steps)
    diffusion = BernoulliDiffusion(scheduler)

    # N=500 (250,000 samples): worst-case standard error ≈ 0.001, 5x inside atol=5e-3.
    N = 500
    x_0 = ones(Float32, N, N) |> Flux.gpu

    t_val = 50
    t = fill(Int32(t_val), N) |> Flux.gpu

    q_sampled = Thesis.q(diffusion, x_0, t) |> Flux.cpu

    ᾱₜ = reference_alphas(total_steps)[t_val]
    expected_prob_keep = 0.5f0 * (1.0f0 + ᾱₜ)

    empirical_prob_keep = mean(q_sampled)

    @info "Theoretical prob_keep:" expected_prob_keep
    @info "Empirical prob_keep:" empirical_prob_keep

    @test size(q_sampled) == (N, N)
    @test eltype(q_sampled) == Float32
    @test all(x -> x == 0.0f0 || x == 1.0f0, q_sampled)
    @test isapprox(empirical_prob_keep, expected_prob_keep, atol=5e-3)
end

@testitem "Test BernoulliDiffusion β schedule (T=1000 unchanged, x_T uniform for other T)" setup=[TestDiffusion] begin
    # T = 1000: exactly DIFUSCO's linear 1e-4 → 2e-2
    @test Thesis.get_betas(1000) ≈ Float32.(range(1f-4, 2f-2, length=1000))
    # q(x_T | x_0) ≈ Uniform for different T thanks to the 1000/T scaling
    for T in (100, 500, 1000)
        @test Thesis.get_alphas(T)[end] < 1f-6
    end
    @test_throws AssertionError Thesis.get_betas(40)
end

@testitem "Test BernoulliDiffusion Posterior Probability exact values" setup=[TestDiffusion] begin
    using Flux
    using Statistics

    total_steps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, total_steps, inference_steps)
    diffusion = BernoulliDiffusion(scheduler)

    x_t = Float32[1.0 0.0; 0.0 1.0] |> Flux.gpu
    x_0_pred = Float32[0.9 0.1; 0.2 0.8] |> Flux.gpu
    t_idx = 50
    s_idx = 40

    p_post = Thesis.posterior_prob(diffusion, x_t, x_0_pred, t_idx, s_idx) |> Flux.cpu

    αs = reference_alphas(total_steps)
    α_bar_t = αs[t_idx]
    α_bar_s = αs[s_idx]
    α_t_s = α_bar_t / α_bar_s

    # q(x_t = 1 | x_s = 1)
    p_t_s = 0.5f0 * (1.0f0 + α_t_s)

    expected = zeros(Float32, 2, 2)
    x_t_cpu = x_t |> Flux.cpu
    x_0_pred_cpu = x_0_pred |> Flux.cpu

    for i in 1:2
        for j in 1:2
            xt_val = x_t_cpu[i, j]
            x0_pred_val = x_0_pred_cpu[i, j]

            term1 = xt_val == 1.0f0 ? p_t_s : (1.0f0 - p_t_s)

            denom_1 = xt_val == 1.0f0 ? 0.5f0 * (1.0f0 + α_bar_t) : 0.5f0 * (1.0f0 - α_bar_t)
            term2 = (term1 * 0.5f0 * (1.0f0 + α_bar_s)) / denom_1

            denom_0 = xt_val == 1.0f0 ? 0.5f0 * (1.0f0 - α_bar_t) : 0.5f0 * (1.0f0 + α_bar_t)
            term3 = (term1 * 0.5f0 * (1.0f0 - α_bar_s)) / denom_0

            expected[i, j] = clamp(x0_pred_val * term2 + (1.0f0 - x0_pred_val) * term3, 0.0f0, 1.0f0)
        end
    end

    @test size(p_post) == size(x_t |> Flux.cpu)
    @test all(x -> 0.0f0 <= x <= 1.0f0, p_post)
    @test p_post ≈ expected atol=1e-5
end

@testitem "Test BernoulliDiffusion Posterior Probability t == s is the identity" setup=[TestDiffusion] begin
    using Flux
    diffusion = BernoulliDiffusion(LinearScheduler(Int, 100, 10))
    x_t = Float32[1.0 0.0 1.0 0.0]
    x_0_pred = Float32[0.1 0.9 0.5 0.3]
    @test Thesis.posterior_prob(diffusion, x_t, x_0_pred, 37, 37) ≈ x_t atol=1e-6
end

@testitem "Test BernoulliDiffusion Posterior Probability boundary case s=0" setup=[TestDiffusion] begin
    using Flux

    total_steps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, total_steps, inference_steps)
    diffusion = BernoulliDiffusion(scheduler)

    x_t = Float32[1.0 0.0] |> Flux.gpu
    x_0_pred = Float32[0.99 0.01] |> Flux.gpu
    t_idx = 10
    s_idx = 0

    p_post = Thesis.posterior_prob(diffusion, x_t, x_0_pred, t_idx, s_idx) |> Flux.cpu

    @info p_post
    @test size(p_post) == size(x_t |> Flux.cpu)
    @test all(x -> 0.0f0 <= x <= 1.0f0, p_post)
end
