using Graphs
using Random

export ERGraphGenerator

"""
    ERGraphGenerator <: GraphGenerator

A `GraphGenerator` wrapper around `NodeSelectionProblems.generate_er_graph`.
Keeps the `GraphGenerator` functor contract required by `RuntimeDataset`
while delegating actual ER generation to the single implementation in
`NodeSelectionProblems.jl` (no duplicated `erdos_renyi` logic).
"""
struct ERGraphGenerator <: GraphGenerator
    min_nodes::Int
    max_nodes::Int
    p::Float64
end

function ERGraphGenerator(; min_nodes::Int=50, max_nodes::Int=100, p::Float64=0.15)
    return ERGraphGenerator(min_nodes, max_nodes, p)
end

function (gen::ERGraphGenerator)()
    v = rand(gen.min_nodes:gen.max_nodes)
    return generate_er_graph(v, gen.p)
end