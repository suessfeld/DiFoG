using TestItems
using DiFoG

@testmodule TestFlowMatching begin
    using Test
    using DiFoG
    using Flux
end

@testitem "Test DiscreteFlowMatching Velocity Field Exact Values" setup=[TestFlowMatching] begin
    using Flux

    T = 100
    scheduler = LinearScheduler(Float32, T, T)
    κ(t) = t
    κ̇(t) = 1.0f0
    dfm = DiscreteFlowMatching(scheduler, 1.0f0, κ, κ̇)

    x_t = Float32[1.0 0.0; 0.0 1.0] |> Flux.gpu
    p1t = Float32[0.9 0.2; 0.1 0.8] |> Flux.gpu
    t = 0.5f0

    u_val = DiFoG.uₜ(dfm, x_t, p1t, t) |> Flux.cpu

    # Eq. 7: u_t = κ̇_t / (1 − κ_t) · (p_{1|t} − x_t); at t = 0.5 the factor is 2
    expected_u = 2.0f0 .* ((p1t |> Flux.cpu) .- (x_t |> Flux.cpu))

    @test size(u_val) == size(x_t |> Flux.cpu)
    @test u_val ≈ expected_u atol=1e-6
end

@testitem "Test DiscreteFlowMatching accepts an Int schedule and converts it" setup=[TestFlowMatching] begin
    dfm = DiscreteFlowMatching(LinearScheduler(1000, 10), 0.1, t -> t^2, t -> 2t)
    @test eltype(typeof(dfm.scheduler)) == Float32
    @test dfm.λ_cons === 0.1f0
    @test dfm.scheduler.steps[1] == 0f0 && dfm.scheduler.steps[end] == 1f0
end

@testitem "Test DiscreteFlowMatching Euler Integration Bounds" setup=[TestFlowMatching] begin
    using Flux

    x_t = Float32[0.8 0.2; 0.1 0.9] |> Flux.gpu
    u_t = Float32[1.0 -1.0; -0.5 0.5] |> Flux.gpu

    # h = 0.4 → unclamped [1.2 -0.2; -0.1 1.1]
    x_next = DiFoG.euler_step(x_t, u_t, 0.2f0, 0.6f0) |> Flux.cpu

    @test x_next ≈ Float32[1.0 0.0; 0.0 1.0] atol=1e-6
end

@testitem "Test DiscreteFlowMatching path sample marginal (Eq. 4–5)" setup=[TestFlowMatching] begin
    using Flux
    using Statistics
    using Random

    M = 250_000
    κ_val = 0.3f0
    x_1 = ones(Float32, 1, M) |> Flux.gpu
    x_src = Float32.(rand!(similar(x_1)) .> 0.5f0)
    u = rand!(similar(x_1))

    x_t = DiFoG._path_sample(fill!(similar(x_1), κ_val), x_1, x_src, u) |> Flux.cpu

    # P(x_t = 1) = κ + (1 − κ) · 1/2 for a uniform binary source
    @test all(x -> x == 0f0 || x == 1f0, x_t)
    @test isapprox(mean(x_t), κ_val + (1f0 - κ_val) * 0.5f0, atol=5e-3)
end

@testitem "Test DiscreteFlowMatching coupled path samples stay on one trajectory (R2)" setup=[TestFlowMatching] begin
    using Random

    M = 10_000
    x_1 = Float32.(rand(M) .> 0.5)'
    x_src = Float32.(rand(M) .> 0.5)'
    u = rand(Float32, 1, M)

    x_a = DiFoG._path_sample(fill(0.2f0, 1, M), x_1, x_src, u)
    x_b = DiFoG._path_sample(fill(0.6f0, 1, M), x_1, x_src, u)

    committed_a = u .< 0.2f0
    # committed at the earlier time ⇒ still equal to x_1 later
    @test all(x_b[committed_a] .== x_1[committed_a])
    # never committed ⇒ keeps its source value at both times (no fresh noise)
    never = u .>= 0.6f0
    @test all(x_a[never] .== x_src[never]) && all(x_b[never] .== x_src[never])
end

@testitem "Test DiscreteFlowMatching consistency segment horizon and Δ" setup=[TestFlowMatching] begin
    # horizon is the smallest point of linspace(0, 1, 20) strictly above τ
    @test DiFoG._segment_horizon(0.0) ≈ 1f0 / 19
    @test DiFoG._segment_horizon(0.5) ≈ 10f0 / 19
    @test DiFoG._segment_horizon(0.999) ≈ 1f0

    T = 1000
    t = Int32[0, 52, 500, 999]
    t_next, horizon = DiFoG._consistency_times(t, T)
    @test all(horizon .> t ./ T)
    @test all(t_next .>= t)
    @test all(t_next ./ T .< horizon)                  # t + Δ stays inside the segment
    @test t_next[1] == 1 && t_next[3] == 501           # Δ = 1e-3 ⇒ one step at T = 1000
    @test t_next[2] == 52                              # 52 is the last step before 1000/19 ≈ 52.6
end

@testitem "Test DiscreteFlowMatching step rules (H1)" setup=[TestFlowMatching] begin
    κ(t) = t^2
    κ̇(t) = 2t
    s = LinearScheduler(Float32, 100, 2)
    euler = DiscreteFlowMatching(s, 0.1f0, κ, κ̇)
    exact = DiscreteFlowMatching(s, 0.1f0, κ, κ̇; sampler=:exact)
    @test euler.sampler === :euler && exact.sampler === :exact
    @test_throws ArgumentError DiscreteFlowMatching(s, 0.1f0, κ, κ̇; sampler=:heun)

    # Euler: c = h·κ̇/(1−κ) → first step from τ = 0 is a no-op (κ̇(0) = 0)
    @test DiFoG._step_coef(euler, 0f0, 0.5f0) ≈ 0f0
    @test DiFoG._step_coef(euler, 0.5f0, 0.75f0) ≈ 0.25f0 * (1f0 / 0.75f0)
    # exact: c = (κ(τ') − κ(τ)) / (1 − κ(τ))
    @test DiFoG._step_coef(exact, 0f0, 0.5f0) ≈ 0.25f0
    @test DiFoG._step_coef(exact, 0.5f0, 0.75f0) ≈ (0.5625f0 - 0.25f0) / 0.75f0
    @test DiFoG._step_coef(exact, 0.5f0, 1f0) ≈ 1f0

    # the Euler step rule reproduces euler_step(x, uₜ(...), τ, τ')
    x = Float32[1 0 1 0]
    p1 = Float32[0.2 0.9 0.6 0.4]
    @test DiFoG._step(x, p1, DiFoG._step_coef(euler, 0.3f0, 0.6f0)) ≈
          DiFoG.euler_step(x, DiFoG.uₜ(euler, x, p1, 0.3f0), 0.3f0, 0.6f0)
end
