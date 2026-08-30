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
    # p → {0,1} everywhere should give entropy → 0 and confidence → 1
    p = Float32[1e-7, 1.0f0 - 1f-7, 1e-7, 1.0f0 - 1f-7]
    mean_entropy, confidence = heatmap_decisiveness(p)

    @test mean_entropy < 1f-4
    @test confidence > 0.999f0
end

@testitem "Test heatmap_decisiveness bounds and clamping" setup=[TestDecodingUtils] begin
    p = Float32[0.0f0, 1.0f0, 0.5f0]
    pseudo_entropy, confidence = heatmap_decisiveness(p)

    @test isfinite(pseudo_entropy)
    @test isfinite(confidence)
    @test 0.0f0 <= confidence <= 1.0f0
    @test 0.0f0 <= pseudo_entropy <= 1.0f0
end

@testitem "Test heatmap_decisiveness bounds and clamping" setup=[TestDecodingUtils] begin
    # Values at exactly 0 and 1 must not produce NaN/Inf due to eps clamping
    p = Float32[0.0f0, 1.0f0, 0.5f0]
    mean_entropy, confidence = heatmap_decisiveness(p)

    @test isfinite(mean_entropy)
    @test isfinite(confidence)
    @test 0.0f0 <= confidence <= 1.0f0
    @test 0.0f0 <= mean_entropy <= Float32(log(2))
end

@testitem "Test correction_distance with no corrections needed" setup=[TestDecodingUtils] begin
    heatmap = Float32[0.9, 0.1, 0.8, 0.2]
    decoded_mask = heatmap .> 0.5f0  # identical to naive threshold decoding

    @test correction_distance(heatmap, decoded_mask) == 0
end

@testitem "Test correction_distance counts flipped nodes" setup=[TestDecodingUtils] begin
    heatmap = Float32[0.9, 0.6, 0.4, 0.1]
    naive = heatmap .> 0.5f0                 # [true, true, false, false]
    decoded_mask = Bool[true, false, true, false]  # greedy corrected 2 nodes

    @test correction_distance(heatmap, decoded_mask) == 2
end

@testitem "Test correction_distance custom threshold" setup=[TestDecodingUtils] begin
    heatmap = Float32[0.3, 0.35, 0.9]
    decoded_mask = Bool[false, false, true]

    # with threshold=0.3 strictly-greater comparison, only 0.35 and 0.9 exceed it
    @test correction_distance(heatmap, decoded_mask; threshold=0.3f0) == 1
end