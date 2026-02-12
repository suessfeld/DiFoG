include("../../utils/data_utils.jl")

using SimpleGraphs
using SimpleGraphAlgorithms
using ChooseOptimizer
using ShowSet
using .DataUtils

DataUtils.load_instance("graph_10_20_1")

function solve_misp(graph::UndirectedGraph)
    mis = maximum_independent_set(graph)
    return mis
    println("Maximum Independent Set: ", mis)
end