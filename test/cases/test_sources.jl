using TestItems
using DiFoG

@testmodule TestSources begin
    using Test
    using DiFoG
    using Flux
    using GraphNeuralNetworks

    export path_graph, triangle, toy_batch, signal

    # Path 1-2-3: the min-degree greedy MIS takes the two end nodes → [1, 0, 1].
    path_graph() = GNNGraph(Int32[1, 2, 2, 3], Int32[2, 1, 3, 2]; num_nodes=3, ndata=(; x=Float32[1 0 1]))
    # Triangle: all degrees equal, the greedy takes node 1 (stable order) → [1, 0, 0].
    triangle() = GNNGraph(Int32[1, 2, 3, 2, 3, 1], Int32[2, 3, 1, 1, 2, 3]; num_nodes=3, ndata=(; x=Float32[0 1 0]))
    toy_batch() = Flux.batch([path_graph(), triangle()])
    signal(y) = 2f0 .* y .- 1f0
end

@testitem "Test GaussianSource statistics and validation" setup=[TestSources] begin
    using Statistics
    g = toy_batch() |> Flux.gpu
    # N = 250,000 samples: standard error of the mean ≈ 0.001, far inside atol = 5e-3.
    x = sample_source(GaussianSource(0.5), g, (500, 500)) |> Flux.cpu
    @test size(x) == (500, 500) && eltype(x) == Float32
    @test isapprox(mean(x), 0f0, atol=5e-3)
    @test isapprox(std(x), 0.5f0, atol=5e-3)
    @test_throws ArgumentError GaussianSource(0)
    @test source_label(GaussianSource(1)) == "Gaussian(sigma=1.0)"
end

@testitem "Test residue of the problems" setup=[TestSources] begin
    # MISP: min-degree greedy independent set (feasible and maximal)
    @test residue(MISPSolver(), path_graph()) == Float32[1, 0, 1]
    @test residue(MISPSolver(), triangle()) == Float32[1, 0, 0]

    # MaxCut: degree-ordered greedy cut; the max-degree node (2 on the path) is on side 1, as in the labels
    r = residue(MaxCutSolver(), path_graph())
    @test r[2] == 1f0 && r == Float32[0, 1, 0]
    @test DiFoG.evaluate(MaxCutSolver(), path_graph(), r) == 2.0

    # MaxDeg: first k nodes in index order, k = number of max-degree nodes (1 on the path, 3 on the triangle)
    @test residue(MaxDegSolver(), path_graph()) == Float32[1, 0, 0]
    @test residue(MaxDegSolver(), triangle()) == Float32[1, 1, 1]
end

@testitem "Test ResidueSource :heuristic keeps the node order of the batch" setup=[TestSources] begin
    g = toy_batch() |> Flux.gpu
    src = ResidueSource(MISPSolver(); kind=:heuristic, σ=0)
    x = sample_source(src, g, (1, 6)) |> Flux.cpu
    @test x == signal(Float32[1 0 1 1 0 0])                    # path residue, then triangle residue

    # a single, unbatched graph works as well
    @test sample_source(src, path_graph() |> Flux.gpu, (1, 3)) |> Flux.cpu == signal(Float32[1 0 1])
end

@testitem "Test ResidueSource noise around the degraded solution" setup=[TestSources] begin
    using Statistics
    g = toy_batch() |> Flux.gpu
    # K = 40,000 rows: every row repeats the residue, so all 240,000 entries share one residue evaluation.
    K = 40_000
    x = sample_source(ResidueSource(MISPSolver(); σ=0.8), g, (K, 6)) |> Flux.cpu
    noise = x .- signal(Float32[1 0 1 1 0 0])
    @test isapprox(mean(noise), 0f0, atol=5e-3)
    @test isapprox(std(noise), 0.8f0, atol=5e-3)
end

@testitem "Test ResidueSource :random is a uniform node subset" setup=[TestSources] begin
    using Statistics
    g = toy_batch() |> Flux.gpu
    x = sample_source(ResidueSource(MISPSolver(); kind=:random, σ=0), g, (500, 500)) |> Flux.cpu
    @test all(v -> v == -1f0 || v == 1f0, x)
    @test isapprox(mean(x .> 0), 0.5, atol=5e-3)               # P(node in X_d) = 1/2 (DISCO)

    @test_throws ArgumentError ResidueSource(MISPSolver(); kind=:greedy)
    @test_throws ArgumentError ResidueSource(MISPSolver(); σ=-1)
    @test source_label(ResidueSource(MISPSolver(); kind=:random)) == "Residue(random, sigma=1.0, MISPSolver)"
end

@testitem "Test RectifiedFlow starts from the residue source" setup=[TestSources] begin
    g = toy_batch()
    y = g.ndata.x
    rf = RectifiedFlow(LinearScheduler(100, 4); source=ResidueSource(MISPSolver(); σ=0))

    # training: without a stored coupling, x_0 comes from the source
    x_t, t, _, aux = DiFoG.corrupt_batch(rf, y, g)
    x_d = signal(Float32[1 0 1 1 0 0])
    @test !aux.coupled && aux.x_0 == x_d
    τ = reshape(Float32.(t[g.graph_indicator]) ./ 100f0, 1, :)
    @test x_t ≈ τ .* signal(y) .+ (1f0 .- τ) .* x_d

    # inference: a model that never moves the state (x̂₁ = x) keeps the degraded solution
    model = (x_, t_, g_) -> log.(((x_ .+ 1f0) ./ 2f0 .+ 1f-6) ./ (1f0 .- (x_ .+ 1f0) ./ 2f0 .+ 1f-6))
    h = DiFoG.sample_heatmap(rf, model, (1, 6), g)
    @test all(0f0 .<= h .<= 1f0)
    @test (h .> 0.5f0) == Bool[1 0 1 1 0 0]
end
