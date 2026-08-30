using TestItems
using Thesis

@testmodule TestGraphGenerators begin
    using Test
    using Thesis
    using Graphs
    export nv, ne, vertices, edges, is_connected, has_edge
end

@testitem "Test BasicGraphGenerator respects node bounds" setup=[TestGraphGenerators] begin
    using Graphs
    gen = BasicGraphGenerator(min_nodes=10, max_nodes=20, min_edges=15, max_edges=30)

    for _ in 1:20
        g = gen()
        @test nv(g) >= 10 && nv(g) <= 20
    end
end

@testitem "Test BasicGraphGenerator respects edge bounds (clamped)" setup=[TestGraphGenerators] begin
    using Graphs
    gen = BasicGraphGenerator(min_nodes=10, max_nodes=10, min_edges=5, max_edges=15)

    for _ in 1:20
        g = gen()
        @test ne(g) >= 5 && ne(g) <= 15
        max_possible = div(nv(g) * (nv(g) - 1), 2)
        @test ne(g) <= max_possible
    end
end

@testitem "Test BasicGraphGenerator produces connected graphs" setup=[TestGraphGenerators] begin
    using Graphs
    # BasicGraphGenerator builds a spanning tree first, so the result must
    # always be connected regardless of how many extra edges are added.
    gen = BasicGraphGenerator(min_nodes=15, max_nodes=25, min_edges=20, max_edges=40)

    for _ in 1:10
        g = gen()
        @test is_connected(g)
    end
end

@testitem "Test BasicGraphGenerator default keyword constructor" setup=[TestGraphGenerators] begin
    using Graphs
    gen = BasicGraphGenerator()
    g = gen()
    @test nv(g) >= 50 && nv(g) <= 100
    @test ne(g) >= 1
end

@testitem "Test BasicGraphGenerator has no self-loops or duplicate edges" setup=[TestGraphGenerators] begin
    using Graphs
    gen = BasicGraphGenerator(min_nodes=10, max_nodes=15, min_edges=15, max_edges=25)

    for _ in 1:10
        g = gen()
        for v in vertices(g)
            @test !has_edge(g, v, v)
        end
        # SimpleGraph inherently disallows duplicate edges, but assert edge
        # count matches the number of unique undirected pairs added.
        seen = Set{Tuple{Int,Int}}()
        for e in edges(g)
            key = (min(src(e), dst(e)), max(src(e), dst(e)))
            @test !(key in seen)
            push!(seen, key)
        end
    end
end

@testitem "Test ERGraphGenerator respects node bounds" setup=[TestGraphGenerators] begin
    using Graphs
    gen = ERGraphGenerator(min_nodes=20, max_nodes=40, p=0.2)

    for _ in 1:20
        g = gen()
        @test nv(g) >= 20 && nv(g) <= 40
    end
end

@testitem "Test ERGraphGenerator handles empty-graph edge case" setup=[TestGraphGenerators] begin
    using Graphs
    # p=0.0 with erdos_renyi could theoretically produce nv(g) == 0 in
    # degenerate cases; the generator falls back to SimpleGraph(v) in that case.
    gen = ERGraphGenerator(min_nodes=5, max_nodes=5, p=0.0)
    g = gen()
    @test nv(g) == 5
    @test ne(g) == 0
end

@testitem "Test ERGraphGenerator default keyword constructor" setup=[TestGraphGenerators] begin
    using Graphs
    gen = ERGraphGenerator()
    g = gen()
    @test nv(g) >= 50 && nv(g) <= 100
end