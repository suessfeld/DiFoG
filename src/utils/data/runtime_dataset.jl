using Flux
using GraphNeuralNetworks
using Graphs

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
    # 1. Generate Graph Topology (Generator uses its own internal params)
    @info "generate"
    g = dataset.generator()
    
    @info "solve"
    # 2. Solve the Problem (Solver uses its own internal params, if any)
    x_0 = dataset.solver(g)
    
    # 3. Extract edge indices for GNNGraph (Bidirectional mapping)
    s = Int32[]
    t = Int32[]
    for edge in edges(g)
        push!(s, Int32(src(edge)))
        push!(t, Int32(dst(edge)))
        push!(s, Int32(dst(edge)))
        push!(t, Int32(src(edge)))
    end
    
    gnn_g = GNNGraph(s, t; num_nodes = nv(g), ndata=(; x=x_0))
    return (gnn_g, "runtime_inst_$i.txt") 
end

function Base.getindex(dataset::RuntimeDataset, inds::AbstractVector{<:Integer})
    tasks = map(inds) do i
        Threads.@spawn dataset[i]
    end
    return fetch.(tasks)
end