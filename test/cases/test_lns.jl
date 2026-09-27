using TestItems
using DiFoG

@testitem "Test LNSSolver construction and forwarding" begin
    using GraphNeuralNetworks

    # defaults = LNS-rand settings of Bresich et al. (d = 5, r = 2000 / 1000, 3 s)
    misp = LNSSolver(MISPSolver())
    maxcut = LNSSolver(MaxCutSolver())
    @test misp.ttime == 3.0 && misp.n_destroy == 5 && misp.n_iter_repair == 2000
    @test maxcut.n_iter_repair == 1000
    @test_throws ArgumentError LNSSolver(MaxCutSolver(); ttime=0)
    @test_throws ArgumentError LNSSolver(MaxCutSolver(); n_destroy=0)
    @test_throws ArgumentError LNSSolver(maxcut)

    # the problem is taken from the wrapped solver
    @test label_symmetric(maxcut) && !label_symmetric(misp)
    @test base_solver(maxcut) isa MaxCutSolver && base_solver(MaxCutSolver()) isa MaxCutSolver
    @test dataset_tag(maxcut) == "lns" && dataset_tag(MaxCutSolver()) == ""

    g = GNNGraph([1, 2, 3, 4, 2, 3, 4, 1], [2, 3, 4, 1, 1, 2, 3, 4]; num_nodes=4)   # 4-cycle
    h = Float32[0.9, 0.2, 0.8, 0.1]
    @test DiFoG.decode_heatmap(misp, g, h) == DiFoG.decode_heatmap(MISPSolver(), g, h)
    @test DiFoG.evaluate(maxcut, g, Bool[1, 0, 1, 0]) == DiFoG.evaluate(MaxCutSolver(), g, Bool[1, 0, 1, 0])
    @test DiFoG.degree_baseline_heatmap(maxcut, g) == DiFoG.degree_baseline_heatmap(MaxCutSolver(), g)
end

@testitem "Test LNSSolver solves a small MaxCut instance" begin
    using Graphs
    using GraphNeuralNetworks

    g = cycle_graph(6)                                   # even cycle: bipartite, maximum cut = 6
    x, gap = LNSSolver(MaxCutSolver(); ttime=0.2)(g)
    @test isnan(gap)                                      # an LNS provides no bound
    @test size(x) == (1, 6) && all(v -> v == 0f0 || v == 1f0, x)
    @test DiFoG.evaluate(MaxCutSolver(), GNNGraph(g), vec(x) .> 0.5f0) == 6.0
    @test x[1, argmax(degree(g))] == 1f0                  # canonical orientation, as the Gurobi labels
end

@testitem "Test LNSSolver solves a small MISP instance" begin
    using Graphs

    g = path_graph(5)                                     # maximum independent set {1, 3, 5}
    x, gap = LNSSolver(MISPSolver(); ttime=0.2)(g)
    @test isnan(gap)
    @test vec(x) == Float32[1, 0, 1, 0, 1]
    @test all(e -> !(x[1, src(e)] == 1f0 && x[1, dst(e)] == 1f0), edges(g))   # independent
end

@testitem "Test create_dataset names LNS datasets with their tag" begin
    using Graphs
    using Random

    dir = mktempdir()
    generator = ERGraphGenerator(min_nodes=8, max_nodes=10, p=0.3)
    path = create_dataset(123, dir, generator, LNSSolver(MaxCutSolver(); ttime=0.05), 3)
    @test basename(path) == "123_lns.csv"                # next to a Gurobi "123.csv" for the same graphs

    lines = readlines(path)
    @test lines[1] == "gap,bitmask" && length(lines) == 4
    rng = Random.Xoshiro(123)                              # same seed → same graphs as the dataset
    for line in lines[2:end]
        gap, bitmask = split(line, ',')
        @test gap == "NaN"
        @test length(bitmask) == nv(generator(rng))
    end
end

@testitem "Test LNSSolver with greedy start" begin
    using Graphs
    using GraphNeuralNetworks
    using Random

    @test_throws ArgumentError LNSSolver(MaxCutSolver(); init=:foo)
    @test LNSSolver(MaxCutSolver()).init === :random                                  # default = LNS-rand
    @test dataset_tag(LNSSolver(MaxCutSolver(); init=:greedy)) == "lnsgreedy"          # own dataset files

    g = ERGraphGenerator(min_nodes=60, max_nodes=60, p=0.2)(Random.Xoshiro(1))
    gnn = GNNGraph(g)
    for base in (MaxCutSolver(), MISPSolver())
        x, gap = LNSSolver(base; ttime=0.2, init=:greedy)(g)                           # `check` inside validates it
        @test isnan(gap)
        greedy = DiFoG.evaluate(base, gnn, DiFoG.greedy_mask(base, g))
        @test DiFoG.evaluate(base, gnn, vec(x) .> 0.5f0) >= greedy                     # never worse than its start
        if base isa MISPSolver
            @test all(e -> !(x[1, src(e)] == 1f0 && x[1, dst(e)] == 1f0), edges(g))     # independent
        end
    end
end
