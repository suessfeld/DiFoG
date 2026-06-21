using Flux
using GraphNeuralNetworks
using Graphs
using .DataUtils: normalize_graph

export RuntimeDataset, GraphGenerator, ProblemSolver

abstract type GraphGenerator end
abstract type ProblemSolver end

struct RuntimeDataset{G<:GraphGenerator, P<:ProblemSolver}
    num_instances::Int
    generator::G
    solver::P
end

Base.length(d::RuntimeDataset) = d.num_instances

function Base.getindex(dataset::RuntimeDataset, i::Int)
    # 1. Generate Graph Topology
    g = dataset.generator()
    
    # 2. Solve the Problem
    x_0 = dataset.solver(g)
    
    # 3. Build the GNNGraph EXACTLY like the old LazyGraphDataset did
    gnn_g = GNNGraph(g, ndata=(; x=x_0))
    
    # 4. Normalize to enforce Int32 edge indices for GPU safety
    return (normalize_graph(gnn_g), "runtime_inst_$i.txt") 
end

function Base.getindex(dataset::RuntimeDataset, inds::AbstractVector{<:Integer})
    tasks = map(inds) do i
        Threads.@spawn dataset[i]
    end
    return fetch.(tasks)
end