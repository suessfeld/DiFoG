using TestItems
using Thesis

@testmodule TestDecodingUtils begin
    using Test
    using Thesis
end

@testitem "Test hamming_distance basic cases" setup=[TestDecodingUtils] begin
    a = Bool[1, 0, 1, 0]
    b = Bool[1, 1, 1, 1]
    @test hamming_distance(a, b) == 2

    @test hamming_distance(a, a) == 0

    c = [1, 0, 1, 0]
    d = [0, 1, 0, 1]
    @test hamming_distance(c, d) == 4
end

@testitem "Test hamming_distance dimension mismatch throws" setup=[TestDecodingUtils] begin
    a = Bool[1, 0, 1]
    b = Bool[1, 0]
    @test_throws AssertionError hamming_distance(a, b)
end

@testitem "Test heatmap_decisiveness fully decisive heatmap" setup=[TestDecodingUtils] begin
    p = Float32[1e-7, 1.0f0 - 1f-7, 1e-7, 1.0f0 - 1f-7]
    pseudo_entropy, confidence = heatmap_decisiveness(p)

    @test pseudo_entropy < 1f-4
    @test confidence > 0.999f0
end

@testitem "Test heatmap_decisiveness exact values at p=0.5 (max uncertainty)" setup=[TestDecodingUtils] begin
    # D̄ = mean(|2p − 1|) = 0 at p = 0.5, so pseudo_entropy = 1 − D̄ = 1.
    p = Float32[0.5f0, 0.5f0, 0.5f0]
    pseudo_entropy, confidence = heatmap_decisiveness(p)

    @test pseudo_entropy ≈ 1.0f0 atol=1f-6
    @test confidence ≈ 0.0f0 atol=1f-6
end

@testitem "Test heatmap_decisiveness bounds and clamping at 0 and 1" setup=[TestDecodingUtils] begin
    p = Float32[0.0f0, 1.0f0, 0.5f0]
    pseudo_entropy, confidence = heatmap_decisiveness(p)

    @test isfinite(pseudo_entropy)
    @test isfinite(confidence)
    @test 0.0f0 <= confidence <= 1.0f0
    @test 0.0f0 <= pseudo_entropy <= 1.0f0
end

@testitem "Test correction_distance with no corrections needed" setup=[TestDecodingUtils] begin
    heatmap = Float32[0.9, 0.1, 0.8, 0.2]
    decoded_mask = heatmap .> 0.5f0

    @test correction_distance(heatmap, decoded_mask) == 0
end

@testitem "Test correction_distance counts flipped nodes" setup=[TestDecodingUtils] begin
    heatmap = Float32[0.9, 0.6, 0.4, 0.1]
    decoded_mask = Bool[true, false, true, false]

    @test correction_distance(heatmap, decoded_mask) == 2
end

@testitem "Test correction_distance custom threshold" setup=[TestDecodingUtils] begin
    heatmap = Float32[0.3, 0.35, 0.9]
    decoded_mask = Bool[false, false, true]

    @test correction_distance(heatmap, decoded_mask; threshold=0.3f0) == 1
end

@testitem "Test heatmap_auc" setup=[TestDecodingUtils] begin
    @test heatmap_auc(Float32[0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) ≈ 1.0
    @test heatmap_auc(Float32[0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) ≈ 0.0
    @test heatmap_auc(fill(0.5f0, 4), [1, 0, 1, 0]) ≈ 0.5           # ties → uninformative
    @test heatmap_auc(Float32[0.9, 0.3, 0.5, 0.1], [1, 0, 0, 1]) ≈ 0.5
    @test isnan(heatmap_auc(Float32[0.1, 0.2], [1, 1]))
end

@testitem "Test baselines on a small graph" setup=[TestDecodingUtils] begin
    using GraphNeuralNetworks
    using Random
    # 4-cycle 1-2-3-4-1: MIS = 2, MaxCut = 4
    s = [1, 2, 3, 4, 2, 3, 4, 1]
    t = [2, 3, 4, 1, 1, 2, 3, 4]
    g = GNNGraph(s, t; num_nodes=4)

    misp = MISPSolver()
    mask = Thesis.decode_heatmap(misp, g, degree_baseline_heatmap(misp, g))
    @test Thesis.evaluate(misp, g, mask) == 2.0

    maxcut = MaxCutSolver()
    @test label_symmetric(maxcut) && !label_symmetric(misp)
    mask = Thesis.decode_heatmap(maxcut, g, degree_baseline_heatmap(maxcut, g))
    @test Thesis.evaluate(maxcut, g, mask) == 4.0

    r = random_baseline_score(maxcut, g, 20; rng=Random.Xoshiro(1))
    @test 0.0 <= r <= 4.0
end

@testitem "Test solution_distance (complement-invariant for MaxCut)" setup=[TestDecodingUtils] begin
    a = [1, 0, 1, 0, 1]
    b = [0, 1, 0, 1, 0]                         # complement of a
    @test solution_distance(a, b) == 5
    @test solution_distance(a, b; symmetric=true) == 0
    @test solution_distance(a, [1, 0, 1, 0, 0]; symmetric=true) == 1
    @test solution_distance(a, [0, 1, 0, 1, 1]; symmetric=true) == 1   # complement differs in one node
end
