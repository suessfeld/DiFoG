using TestItems
using DiFoG

@testmodule TestDataUtils begin
    using Test
    using DiFoG
    using GraphNeuralNetworks
    export GNNGraph, edge_index
end

@testitem "Test normalize_graph converts edge indices to Int32" setup=[TestDataUtils] begin
    using GraphNeuralNetworks
    s = [1, 2, 3]
    t = [2, 3, 1]
    g = GNNGraph(s, t; num_nodes=3, ndata=(; x=rand(Float32, 1, 3)))

    g_norm = DiFoG.normalize_graph(g)

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

    g_norm = DiFoG.normalize_graph(g)

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

    g_norm = DiFoG.normalize_graph(g)

    @test g_norm.ndata.x == g.ndata.x
end
@testitem "Test RuntimeDataset shuffling (new order per pass, same instances)" begin
    using GraphNeuralNetworks
    using Random

    dir = mktempdir()
    generator = ERGraphGenerator(min_nodes=12, max_nodes=18, p=0.3)
    solver = LNSSolver(MaxCutSolver(); ttime=-1, titer=20, n_iter_repair=1)
    n = 12
    seq = RuntimeDataset(n, generator, solver, 11, dir)
    seq_items = [seq[i] for i in 1:n]

    Random.seed!(1)
    shuf = RuntimeDataset(n, generator, solver, 11, dir; shuffle=true)
    pass1 = [shuf[i] for i in 1:n]
    DiFoG.reset!(shuf)
    pass2 = [shuf[i] for i in 1:n]

    by_name = Dict(name => g for (g, name) in seq_items)
    for pass in (pass1, pass2)
        @test sort(last.(pass)) == sort(last.(seq_items))                  # the same instances
        for (g, name) in pass                                              # and exactly the same graphs and labels
            @test g.num_nodes == by_name[name].num_nodes && g.num_edges == by_name[name].num_edges
            @test g.ndata.x == by_name[name].ndata.x
        end
    end
    @test last.(pass1) != last.(pass2)                                     # a new order per pass
    @test last.(pass1) != last.(seq_items)

    # reproducible under the same global seed
    Random.seed!(1)
    shuf2 = RuntimeDataset(n, generator, solver, 11, dir; shuffle=true)
    @test last.([shuf2[i] for i in 1:n]) == last.(pass1)
end
