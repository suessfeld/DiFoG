# src/utils/data/runtime_dataset.jl
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
    buffer::Channel{Tuple{GNNGraph,String}}
end

"""
    RuntimeDataset(num_instances, generator, solver; max_concurrent_solves=2, buffer_capacity=4*max_concurrent_solves)

Dataset that is solved during runtime.
TODO Docs
"""
function RuntimeDataset(num_instances::Int, generator::G, solver::P;
                         max_concurrent_solves::Int=2,
                         buffer_capacity::Int=3 * max_concurrent_solves) where {G<:GraphGenerator, P<:ProblemSolver}
    buffer = Channel{Tuple{GNNGraph,String}}(buffer_capacity)

    for _ in 1:max_concurrent_solves
        Threads.@spawn begin
            n = 0
            while true
                try
                    g = generator()
                    x_0 = solver(g)
                    gnn_g = GNNGraph(g, ndata=(; x=x_0))
                
                    put!(buffer, (normalize_graph(gnn_g), "runtime_inst_$(n).txt"))
                    n += 1
                catch e
                    # Ensures the background task doesn't die silently! This happened sometimes at the cluster - I don't know if it is an Gurobi issue...
                    @error "Background dataset worker crashed! Logging and resuming..." exception=(e, catch_backtrace())
                    sleep(1)
                end
            end
        end
    end

    return RuntimeDataset(num_instances, generator, solver, buffer)
end

Base.length(d::RuntimeDataset) = d.num_instances

Base.getindex(dataset::RuntimeDataset, ::Int) = take!(dataset.buffer)

function Base.getindex(dataset::RuntimeDataset, inds::AbstractVector{<:Integer})
    return [take!(dataset.buffer) for _ in inds]
end