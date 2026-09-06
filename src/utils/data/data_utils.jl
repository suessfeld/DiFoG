# src/utils/data/data_utils.jl
#
# Data utilities for normalizing graph formats and ensuring correct data types for GPU compatibility.

module DataUtils

using Flux
using GraphNeuralNetworks

export normalize_graph

"""
    normalize_graph(g::GNNGraph)

Normalizes an input graph so it uses correct types for GPU usage.

Parameters:
    'g': the graph that will be normalized
"""
function normalize_graph(g::GNNGraph)
    s, t = edge_index(g)
    return GNNGraph(
        Int32.(s), Int32.(t);
        num_nodes = g.num_nodes,
        ndata     = g.ndata,
        edata     = g.edata,
        gdata     = g.gdata,
    )
end

end