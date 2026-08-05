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
    
    @test length(scheduler) == inference_steps
    @test eltype(typeof(scheduler)) == Int
    @test get_num_timesteps(scheduler) == num_timesteps
    
    steps = collect(scheduler)
    @test typeof(steps) == Vector{Int}
    @test length(steps) == inference_steps
    
    c = 1.0 / inference_steps
    expected_steps = [max(1, floor(Int, cos((1.0 - c * i) * (π / 2.0)) * num_timesteps)) for i in 1:inference_steps]
    
    @test steps == expected_steps
    @test all(x -> x >= 1 && x <= num_timesteps, steps)
end

@testitem "Test Continuous Cosine Scheduler" setup=[TestSchedulers] begin
    num_timesteps = 100
    inference_steps = 10
    
    scheduler = CosineScheduler(Float32, num_timesteps, inference_steps)
    
    @test length(scheduler) == inference_steps
    @test eltype(typeof(scheduler)) == Float32
    @test get_num_timesteps(scheduler) == num_timesteps
    
    steps = collect(scheduler)
    @test typeof(steps) == Vector{Float32}
    @test length(steps) == inference_steps
    
    c = 1.0 / inference_steps
    steps_int = [max(1, floor(Int, cos((1.0 - c * i) * (π / 2.0)) * num_timesteps)) for i in 1:inference_steps]
    expected_steps = Float32.(steps_int) ./ Float32(num_timesteps)
    
    @test steps ≈ expected_steps
    @test all(x -> 0.0f0 <= x <= 1.0f0, steps)
end