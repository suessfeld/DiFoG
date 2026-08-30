using TestItems
using Thesis

@testmodule TestSchedulers begin
    using Test
    using Thesis
end

@testitem "Test Discrete Cosine Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10
    
    scheduler = CosineScheduler(Int, num_timesteps, inference_steps)
    
    @test length(scheduler) == inference_steps - 1
    @test eltype(typeof(scheduler)) == Int
    @test scheduler.total_steps == num_timesteps
    
    steps = collect(scheduler)
    @test typeof(steps) == Vector{Int}
    @test length(steps) == inference_steps
    
    c = 1.0 / inference_steps
    expected_steps = [max(1, floor(Int, cos((1.0 - c * i) * (π / 2.0)) * num_timesteps)) for i in 1:inference_steps]
    
    @info steps
    @test steps == expected_steps
    @test all(x -> x >= 1 && x <= num_timesteps, steps)
end

@testitem "Test Continuous Cosine Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10
    
    scheduler = CosineScheduler(Float32, num_timesteps, inference_steps)
    
    @test length(scheduler) == inference_steps - 1
    @test eltype(typeof(scheduler)) == Float32
    @test scheduler.total_steps == num_timesteps
    
    steps = collect(scheduler)
    @test typeof(steps) == Vector{Float32}
    @test length(steps) == inference_steps
    
    c = 1.0 / inference_steps
    steps_int = [max(1, floor(Int, cos((1.0 - c * i) * (π / 2.0)) * num_timesteps)) for i in 1:inference_steps]
    expected_steps = Float32.(steps_int) ./ Float32(num_timesteps)
    
    @info steps
    @test steps ≈ expected_steps
    @test all(x -> 0.0f0 <= x <= 1.0f0, steps)
end

@testitem "Test Discrete Linear Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10

    scheduler = LinearScheduler(Int, num_timesteps, inference_steps)

    @test length(scheduler) == inference_steps - 1
    @test eltype(typeof(scheduler)) == Int
    @test scheduler.total_steps == num_timesteps

    steps = collect(scheduler)
    @test typeof(steps) == Vector{Int}
    @test length(steps) == inference_steps

    c = Float64(num_timesteps) / Float64(inference_steps)
    expected_steps = [max(1, floor(Int, c * i)) for i in 1:inference_steps]

    @info steps
    @test steps == expected_steps
    @test all(x -> x >= 1 && x <= num_timesteps, steps)
    @test issorted(steps)
end

@testitem "Test Continuous Linear Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10

    scheduler = LinearScheduler(Float32, num_timesteps, inference_steps)

    @test length(scheduler) == inference_steps - 1
    @test eltype(typeof(scheduler)) == Float32
    @test scheduler.total_steps == num_timesteps

    steps = collect(scheduler)
    @test typeof(steps) == Vector{Float32}
    @test length(steps) == inference_steps

    c = Float64(num_timesteps) / Float64(inference_steps)
    steps_int = [max(1, floor(Int, c * i)) for i in 1:inference_steps]
    expected_steps = Float32.(steps_int) ./ Float32(num_timesteps)

    @info steps
    @test steps ≈ expected_steps
    @test all(x -> 0.0f0 <= x <= 1.0f0, steps)
end

@testitem "Test LinearScheduler with inference_steps >= total_steps" setup=[TestSchedulers] begin
    num_timesteps = 20
    inference_steps = 20

    scheduler_int = LinearScheduler(Int, num_timesteps, inference_steps)
    steps_int = collect(scheduler_int)
    @test steps_int == collect(1:num_timesteps)

    scheduler_float = LinearScheduler(Float32, num_timesteps, inference_steps)
    steps_float = collect(scheduler_float)
    @test steps_float ≈ collect(1:num_timesteps) ./ Float32(num_timesteps)
end

@testitem "Test LinearScheduler Base.convert roundtrip" setup=[TestSchedulers] begin
    num_timesteps = 50
    inference_steps = 5
 
    s_int = LinearScheduler(Int, num_timesteps, inference_steps)
    s_float = convert(Scheduler{Float32}, s_int)
 
    @test eltype(typeof(s_float)) == Float32
    @test length(s_float) == length(s_int)
    @test s_float.total_steps == s_int.total_steps
end
 
@testitem "Test LinearScheduler iteration order (descending)" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10
    scheduler = LinearScheduler(Int, num_timesteps, inference_steps)
 
    collected_via_iteration = Int[]
    for τ in scheduler
        push!(collected_via_iteration, τ)
    end
 
    @test length(collected_via_iteration) == length(scheduler)
    @test issorted(collected_via_iteration, rev=true)

    @test collected_via_iteration == scheduler.steps[end-1:-1:1]
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
    
    @test length(scheduler_float) == inference_steps
    @test eltype(typeof(scheduler_float)) == Float32
    
    steps_float = collect(scheduler_float)
    @test length(steps_float) == inference_steps + 1
    @test all(x -> 0.0f0 <= x <= 1.0f0, steps_float)
    
    scheduler_int = convert(Scheduler{Int}, scheduler_float)
    @test eltype(typeof(scheduler_int)) == Int
    
    steps_int = collect(scheduler_int)
    @info steps_int
    @info steps_float
    @test length(steps_int) == inference_steps + 1
    @test all(x -> 1 <= x <= total_steps, steps_int)
end

@testitem "Test ATSS Scheduler Lambda Inverse Logic" setup=[TestSchedulers] begin
    # Test the internal bisection search used for log-interpolation ratio inversion
    κ(t) = t
    
    # Given lambda = log(0.5 / 0.5) = 0.0, the inverse should be t = 0.5
    λ_target = 0.0f0 
    t_recovered = Thesis._lambda_inverse(κ, λ_target, tol=1f-5)
    
    @test t_recovered ≈ 0.5f0 atol=1f-4
end
 