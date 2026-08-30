using TestItems
using Thesis

@testmodule TestDataUtils begin
    using Test
    using Thesis
    using GraphNeuralNetworks
    export GNNGraph, edge_index
end

@testitem "Test normalize_graph converts edge indices to Int32" setup=[TestDataUtils] begin
    using GraphNeuralNetworks
    s = [1, 2, 3]
    t = [2, 3, 1]
    g = GNNGraph(s, t; num_nodes=3, ndata=(; x=rand(Float32, 1, 3)))

    g_norm = Thesis.normalize_graph(g)

    src_norm, dst_norm = edge_index(g_norm)
    @test eltype(src_norm) == Int32
    @test eltype(dst_norm) == Int32
end

@testitem "Test normalize_graph preserves graph structure" setup=[TestDataUtils] begin
    using GraphNeuralNetworks
    s = [1, 2, 3, 4]
    t = [2, 3, 4, 1]
    x0 = rand(Float32, 1, 4)
    g = GNNGraph(s, t; num_nodes=4, ndata=(; x=x0))

    g_norm = Thesis.normalize_graph(g)

    @test g_norm.num_nodes == g.num_nodes
    @test g_norm.num_edges == g.num_edges

    src_orig, dst_orig = edge_index(g)
    src_norm, dst_norm = edge_index(g_norm)
    @test Int.(src_norm) == src_orig
    @test Int.(dst_norm) == dst_orig
end

@testitem "Test normalize_graph preserves node features" setup=[TestDataUtils] begin
    using GraphNeuralNetworks
    s = [1, 2]
    t = [2, 1]
    x0 = Float32[0.1 0.9]
    g = GNNGraph(s, t; num_nodes=2, ndata=(; x=x0))

    g_norm = Thesis.normalize_graph(g)

    @test g_norm.ndata.x == g.ndata.x
end