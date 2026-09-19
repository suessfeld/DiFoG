@testitem "Test DiscreteFlowMatching Forward Transition Statistical Distribution" setup=[TestFlowMatching] begin
    using Flux
    using Statistics

    T = 100
    scheduler = LinearScheduler(Float32, T, T)
    κ(t) = t
    κ̇(t) = 1.0f0
    dfm = DiscreteFlowMatching(scheduler, 1.0f0, κ, κ̇)

    N = 500

    # x_1 is the target solution (all 1s for testing)
    x_1 = ones(Float32, N, N) |> Flux.gpu

    # x_t is entirely uncommitted noise (all 0s, strictly opposite to x_1)
    x_t = zeros(Float32, N, N) |> Flux.gpu

    # B1 fix: forward_transition_step now takes the ground-truth commit mask instead of inferring
    # it via x_t .== x_1. x_t is deliberately the exact complement of x_1 here, so every entry is
    # genuinely uncommitted — the correct mask is all-false, matching what the old x_t==x_1 proxy
    # happened to (coincidentally, and only because of this specific test setup) also produce.
    committed = falses(N, N) |> Flux.gpu

    # Define a transition interval
    t = fill(0.2f0, N, N) |> Flux.gpu
    t_next = fill(0.5f0, N, N) |> Flux.gpu

    x_next = Thesis.forward_transition_step(dfm, x_t, x_1, t, t_next, committed) |> Flux.cpu

    # ---------------------------------------------------------
    # Theoretical Expected Distribution
    # ---------------------------------------------------------
    # Probability of an uncommitted node permanently committing to the target x_1 
    # over the interval [t, t_next] is given by (κ(t_next) - κ(t)) / (1 - κ(t))
    p_commit = (0.5f0 - 0.2f0) / (1.0f0 - 0.2f0) # 0.375
    
    # Nodes that do not commit (1 - p_commit) are replaced with uniform noise (0.5 prob of matching 1)
    expected_prob_1 = p_commit + (1.0f0 - p_commit) * 0.5f0 # 0.375 + 0.3125 = 0.6875
    
    # ---------------------------------------------------------
    # Empirical Validation
    # ---------------------------------------------------------
    empirical_prob_1 = mean(x_next)
    
    @info "Theoretical expected ratio of 1s:" expected_prob_1
    @info "Empirical ratio of 1s:" empirical_prob_1
    
    @test size(x_next) == (N, N)
    
    # Ensure it only produces valid categorical {0, 1} states required for discrete trajectories
    @test all(x -> x == 0.0f0 || x == 1.0f0, x_next)
    
    # Assert the sampled distribution matches the paper's theoretical equation within a strict tolerance
    @test isapprox(empirical_prob_1, expected_prob_1, atol=5e-3)
end