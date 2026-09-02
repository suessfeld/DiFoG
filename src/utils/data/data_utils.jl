#GR TODO: Finde es übertrieben hierfür ein eigenes Modul zu haben, kann in NodeSelection Problems integriert werden.

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