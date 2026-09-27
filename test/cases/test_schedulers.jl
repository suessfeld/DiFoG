using TestItems
using DiFoG

@testmodule TestSchedulers begin
    using Test
    using DiFoG
end

@testitem "Test Discrete Cosine Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10

    scheduler = CosineScheduler(Int, num_timesteps, inference_steps)

    # schedule is [0, …ascending interior…, total_steps]; consumers drop the self-known endpoint via
    # `Iterators.drop(..., 1)`, so `inference_steps` transitions are performed.
    @test length(scheduler) == inference_steps + 1
    @test eltype(typeof(scheduler)) == Int
    @test scheduler.total_steps == num_timesteps

    steps = collect(scheduler)
    @test typeof(steps) == Vector{Int}
    @test length(steps) == inference_steps + 1
    @test steps[1] == 0
    @test steps[end] == num_timesteps
    @test issorted(steps)
    @test collect(scheduler) == [x for x in scheduler]

    # C3: Int steps are diffusion times t = T − p of the progress points p_i = ⌊cos((1−ci)π/2)·T⌋
    c = 1.0 / inference_steps
    progress = vcat(0, [max(1, floor(Int, cos((1.0 - c * i) * (π / 2.0)) * num_timesteps)) for i in 1:inference_steps])
    expected_steps = sort(num_timesteps .- progress)

    @info steps
    @test steps == expected_steps
    @test all(x -> x >= 0 && x <= num_timesteps, steps)
    # "more steps in the low-noise regime": the gaps near t = 0 are smaller than near t = T
    @test steps[2] - steps[1] < steps[end] - steps[end-1]
end

@testitem "Test collect returns a copy (R23)" setup=[TestSchedulers] begin
    scheduler = LinearScheduler(Int, 100, 10)
    steps = collect(scheduler)
    steps[2] = -1
    @test scheduler.steps[2] != -1
end

@testitem "Test Continuous Cosine Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10

    scheduler = CosineScheduler(Float32, num_timesteps, inference_steps)

    @test length(scheduler) == inference_steps + 1
    @test eltype(typeof(scheduler)) == Float32
    @test scheduler.total_steps == num_timesteps

    steps = collect(scheduler)
    @test typeof(steps) == Vector{Float32}
    @test length(steps) == inference_steps + 1
    @test steps[1] == 0.0f0
    @test steps[end] == 1.0f0
    @test issorted(steps)
    @test collect(scheduler) == [x for x in scheduler]

    c = 1.0 / inference_steps
    steps_int = [max(1, floor(Int, cos((1.0 - c * i) * (π / 2.0)) * num_timesteps)) for i in 1:inference_steps]
    expected_steps = vcat(0.0f0, Float32.(steps_int) ./ Float32(num_timesteps))

    @info steps
    @test steps ≈ expected_steps
    @test all(x -> 0.0f0 <= x <= 1.0f0, steps)
end

@testitem "Test Discrete Linear Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10

    scheduler = LinearScheduler(Int, num_timesteps, inference_steps)

    @test length(scheduler) == inference_steps + 1
    @test eltype(typeof(scheduler)) == Int
    @test scheduler.total_steps == num_timesteps

    steps = collect(scheduler)
    @test typeof(steps) == Vector{Int}
    @test length(steps) == inference_steps + 1
    @test steps[1] == 0
    @test steps[end] == num_timesteps
    @test issorted(steps)
    @test collect(scheduler) == [x for x in scheduler]

    c = Float64(num_timesteps) / Float64(inference_steps)
    expected_steps = vcat(0, [max(1, floor(Int, c * i)) for i in 1:inference_steps])

    @info steps
    @test steps == expected_steps
    @test all(x -> x >= 0 && x <= num_timesteps, steps)
end

@testitem "Test Continuous Linear Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10

    scheduler = LinearScheduler(Float32, num_timesteps, inference_steps)

    @test length(scheduler) == inference_steps + 1
    @test eltype(typeof(scheduler)) == Float32
    @test scheduler.total_steps == num_timesteps

    steps = collect(scheduler)
    @test typeof(steps) == Vector{Float32}
    @test length(steps) == inference_steps + 1
    @test steps[1] == 0.0f0
    @test steps[end] == 1.0f0
    @test issorted(steps)
    @test collect(scheduler) == [x for x in scheduler]

    c = Float64(num_timesteps) / Float64(inference_steps)
    steps_int = [max(1, floor(Int, c * i)) for i in 1:inference_steps]
    expected_steps = vcat(0.0f0, Float32.(steps_int) ./ Float32(num_timesteps))

    @info steps
    @test steps ≈ expected_steps
    @test all(x -> 0.0f0 <= x <= 1.0f0, steps)
end

@testitem "Test LinearScheduler with inference_steps >= total_steps" setup=[TestSchedulers] begin
    num_timesteps = 20
    inference_steps = 20

    scheduler_int = LinearScheduler(Int, num_timesteps, inference_steps)
    steps_int = collect(scheduler_int)
    @test steps_int == collect(0:num_timesteps)

    scheduler_float = LinearScheduler(Float32, num_timesteps, inference_steps)
    steps_float = collect(scheduler_float)
    @test steps_float ≈ collect(0:num_timesteps) ./ Float32(num_timesteps)
end

@testitem "Test LinearScheduler Base.convert roundtrip" setup=[TestSchedulers] begin
    num_timesteps = 50
    inference_steps = 5

    s_int = LinearScheduler(Int, num_timesteps, inference_steps)
    s_float = convert(TimeSchedule{Float32}, s_int)

    @test eltype(typeof(s_float)) == Float32
    @test length(s_float) == length(s_int)
    @test s_float.total_steps == s_int.total_steps
end

@testitem "Test LinearScheduler iteration order (ascending default, descending via reverse)" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, num_timesteps, inference_steps)

    collected_forward = Int[]
    for τ in scheduler
        push!(collected_forward, τ)
    end

    @test length(collected_forward) == length(scheduler)
    @test issorted(collected_forward)
    @test collected_forward == scheduler.steps
    @test collected_forward == collect(scheduler)

    collected_reverse = Int[]
    for τ in Iterators.reverse(scheduler)
        push!(collected_reverse, τ)
    end

    @test length(collected_reverse) == length(scheduler)
    @test issorted(collected_reverse, rev=true)
    @test collected_reverse == reverse(scheduler.steps)
end

@testitem "Test ATSS Scheduler Generation and Bounds" setup=[TestSchedulers] begin
    using Graphs

    total_steps = 100
    inference_steps = 10

    generator = BasicGraphGenerator(min_nodes=10, max_nodes=15, min_edges=20, max_edges=30)
    solver = MaxDegSolver()

    κ(t) = t
    κ̇(t) = 1.0f0

    scheduler_float = ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; atss_p=1.0f0)

    @test length(scheduler_float) == inference_steps + 1
    @test eltype(typeof(scheduler_float)) == Float32

    steps_float = collect(scheduler_float)
    @test length(steps_float) == inference_steps + 1
    @test steps_float[1] == 0.0f0
    @test steps_float[end] == 1.0f0
    @test issorted(steps_float)
    @test all(x -> 0.0f0 <= x <= 1.0f0, steps_float)

    scheduler_int = convert(TimeSchedule{Int}, scheduler_float)
    @test eltype(typeof(scheduler_int)) == Int

    steps_int = collect(scheduler_int)
    @info steps_int
    @info steps_float
    @test length(steps_int) == inference_steps + 1
    @test steps_int[1] == 0
    @test steps_int[end] == total_steps
    @test issorted(steps_int)
    @test allunique(steps_int)
    @test all(x -> 0 <= x <= total_steps, steps_int)
end

@testitem "Test ATSS on the paths of DDPM and Rectified Flow" setup=[TestSchedulers] begin
    using Graphs
    using Logging

    total_steps = 100
    inference_steps = 5
    generator = BasicGraphGenerator(min_nodes=10, max_nodes=15, min_edges=20, max_edges=30)
    solver = MaxDegSolver()
    κ(t) = t^2
    κ̇(t) = 2t

    # κ of the Bernoulli forward process: ᾱ at diffusion time T(1 − τ), from ≈ 0 (noise) to 1 (data)
    κ_b = DiFoG._bernoulli_kappa(total_steps)
    @test κ_b(1f0) ≈ 1f0
    @test κ_b(0f0) ≈ DiFoG.get_alphas(total_steps)[end]
    @test issorted([κ_b(τ) for τ in 0f0:0.01f0:1f0])

    for path in (:diffusion, :rectified_flow)
        s = ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; path=path)
        @test s.path === path
        @test length(s) == inference_steps + 1
        @test s.steps[1] == 0f0 && s.steps[end] == 1f0 && issorted(s.steps)
        @test convert(TimeSchedule{Int}, s).path === path          # the conversion keeps the path
    end
    @test ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇).path === :flow_matching
    @test_throws ArgumentError ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; path=:foo)

    # a method only warns when the grid was fitted on another method's path
    s_ddpm = ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; path=:diffusion)
    @test_logs min_level=Logging.Warn BernoulliDiffusion(s_ddpm)
    @test_logs (:warn,) match_mode=:any DiscreteFlowMatching(s_ddpm, 0.1f0, κ, κ̇)
    s_rf = ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; path=:rectified_flow)
    @test_logs min_level=Logging.Warn RectifiedFlow(s_rf)
end

@testitem "Test ATSS Scheduler Lambda Inverse Logic" setup=[TestSchedulers] begin
    κ(t) = t

    # λ = log(0.5 / 0.5) = 0 ⇒ t = 0.5
    t_recovered = DiFoG._lambda_inverse(κ, 0.0f0, tol=1f-5)

    @test t_recovered ≈ 0.5f0 atol=1f-4
end

@testitem "Test schedules plug into both methods" setup=[TestSchedulers] begin
    bd = BernoulliDiffusion(CosineScheduler(Float32, 1000, 10))     # Float32 → Int
    @test eltype(typeof(bd.scheduler)) == Int
    @test bd.scheduler.steps[end] == 1000
    dfm = DiscreteFlowMatching(CosineScheduler(1000, 10), 0.1f0, t -> t^2, t -> 2t)   # Int → Float32
    @test eltype(typeof(dfm.scheduler)) == Float32
end

@testitem "Test time convention: Int steps mirror Float progress (C3)" setup=[TestSchedulers] begin
    T, M = 1000, 7   # non-integer c = T/M, so linear is not trivially symmetric
    for S in (LinearScheduler, CosineScheduler)
        s_int = S(Int, T, M)
        s_float = S(Float32, T, M)
        @test s_int.steps == sort(round.(Int, T .* (1f0 .- s_float.steps)))
    end
    # cosine is dense near the DATA end for both methods
    cf = collect(CosineScheduler(Float32, T, M))
    @test cf[end] - cf[end-1] < cf[2] - cf[1]
    ci = collect(CosineScheduler(Int, T, M))
    @test ci[2] - ci[1] < ci[end] - ci[end-1]
end

@testitem "Test ATSS Int conversion maps progress to diffusion time (C3)" setup=[TestSchedulers] begin
    s_float = DiFoG.ATSSScheduler{Float32}(100, 3, Float32[0, 0.1, 0.95, 1], t -> t, t -> 1f0, 0.5f0)
    s_int = convert(TimeSchedule{Int}, s_float)
    @test s_int.steps == [0, 5, 90, 100]              # t = T(1 − τ), sorted ascending
    back = convert(TimeSchedule{Float32}, s_int)
    @test back.steps ≈ Float32[0, 0.1, 0.95, 1]
end
