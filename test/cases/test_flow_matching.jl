# cases/test_flow_matching.jl
using TestItems

@testmodule TestFlowMatching begin
    using Test
    using Thesis
end

@testitem "Test DiscreteFlowMatching Velocity Field Exact Values" setup=[TestFlowMatching] begin
    using Flux

    T = 100
    scheduler = LinearScheduler(Float32, T, T)
    
    # Simple linear interpolation for the probability path[cite: 4]
    κ(t) = t
    κ̇(t) = 1.0f0
    dfm = DiscreteFlowMatching(scheduler, 1.0f0, κ, κ̇)

    # Move mock inputs to GPU
    x_t = Float32[1.0 0.0; 0.0 1.0] |> Flux.gpu
    p1t = Float32[0.9 0.2; 0.1 0.8] |> Flux.gpu
    t = 0.5f0
    
    u_val = Thesis.uₜ(dfm, x_t, p1t, t) |> Flux.cpu
    
    # ---------------------------------------------------------
    # Deterministic Ground Truth Calculation (Eq 7)[cite: 4]
    # ---------------------------------------------------------
    # u_t = (κ̇_t / (1.0 - κ_t)) * (p1t - x_t)[cite: 4]
    # At t = 0.5: (1.0 / (1.0 - 0.5)) = 2.0
    expected_u = 2.0f0 .* ( (p1t |> Flux.cpu) .- (x_t |> Flux.cpu) )
    
    @test size(u_val) == size(x_t |> Flux.cpu)
    @test u_val ≈ expected_u atol=1e-6
end

@testitem "Test DiscreteFlowMatching Forward Transition Statistical Distribution" setup=[TestFlowMatching] begin
    using Flux
    using Statistics

    T = 100
    scheduler = LinearScheduler(Float32, T, T)
    κ(t) = t
    κ̇(t) = 1.0f0
    dfm = DiscreteFlowMatching(scheduler, 1.0f0, κ, κ̇)

    # 1. Setup a massive tensor to allow the Law of Large Numbers to converge
    N = 5000
    
    # x_1 is the target solution (all 1s for testing)
    x_1 = ones(Float32, N, N) |> Flux.gpu
    
    # x_t is entirely uncommitted noise (all 0s, strictly opposite to x_1)
    x_t = zeros(Float32, N, N) |> Flux.gpu
    
    # Define a transition interval
    t = fill(0.2f0, N, N) |> Flux.gpu
    t_next = fill(0.5f0, N, N) |> Flux.gpu
    
    x_next = Thesis.forward_transition_step(dfm, x_t, x_1, t, t_next) |> Flux.cpu
    
    # ---------------------------------------------------------
    # Theoretical Expected Distribution[cite: 4]
    # ---------------------------------------------------------
    # Probability of an uncommitted node permanently committing to the target x_1 
    # over the interval [t, t_next] is given by (κ(t_next) - κ(t)) / (1 - κ(t))
    p_commit = (0.5f0 - 0.2f0) / (1.0f0 - 0.2f0) # 0.375
    
    # Nodes that do not commit (1 - p_commit) are replaced with uniform noise (0.5 prob of matching 1)[cite: 4]
    expected_prob_1 = p_commit + (1.0f0 - p_commit) * 0.5f0 # 0.375 + 0.3125 = 0.6875
    
    # ---------------------------------------------------------
    # Empirical Validation
    # ---------------------------------------------------------
    empirical_prob_1 = mean(x_next)
    
    @info "Theoretical expected ratio of 1s:" expected_prob_1
    @info "Empirical ratio of 1s:" empirical_prob_1
    
    @test size(x_next) == (N, N)
    
    # Ensure it only produces valid categorical {0, 1} states required for discrete trajectories[cite: 4]
    @test all(x -> x == 0.0f0 || x == 1.0f0, x_next)
    
    # Assert the sampled distribution matches the paper's theoretical equation within a strict tolerance
    @test isapprox(empirical_prob_1, expected_prob_1, atol=5e-3)
end

@testitem "Test DiscreteFlowMatching Euler Integration Bounds" setup=[TestFlowMatching] begin
    using Flux
    
    # ---------------------------------------------------------
    # Deterministic Bounds Check (Eq 6 Context)[cite: 4]
    # ---------------------------------------------------------
    x_t = Float32[0.8 0.2; 0.1 0.9] |> Flux.gpu
    
    # Provide strong velocities designed to push the probability state out of [0, 1] bounds[cite: 4]
    u_t = Float32[1.0 -1.0; -0.5 0.5] |> Flux.gpu
    
    τ = 0.2f0
    τ_next = 0.6f0
    
    # Expected Integration:
    # h = 0.4
    # Unclamped expected:
    # [0.8 + 0.4 = 1.2,  0.2 - 0.4 = -0.2]
    # [0.1 - 0.2 = -0.1, 0.9 + 0.2 = 1.1]
    
    x_next = Thesis.euler_step(x_t, u_t, τ, τ_next) |> Flux.cpu
    
    expected = Float32[1.0 0.0; 0.0 1.0]
    
    @test x_next ≈ expected atol=1e-6
end