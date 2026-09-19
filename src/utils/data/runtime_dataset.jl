# src/utils/data/runtime_dataset.jl
#
# Implements a seeded, asynchronous dataset generator that yields graph instances and their exact
# solver solutions.
#
# Ensures `<filepath>/<seed>.csv` exists via `create_dataset` (creating it if needed, reusing it
# otherwise), then replays it: regenerates each graph from a fresh `Xoshiro(seed)` (matching
# `create_dataset`'s generation order) and reads its already-solved label from the CSV instead of
# solving it again. This is necessarily sequential — the RNG stream must be consumed in order — so
# there is no concurrent-solving mode.
#
# The producer task is finite: it stops after exactly `num_instances` items (a full replay of the
# CSV), so a `RuntimeDataset` only supports being iterated ONCE end-to-end by default — e.g. one
# training epoch. Call `reset!` to respawn the producer and support another full pass (see
# `train_diffusion_model!` in `diffusion_model.jl`, which calls this once per epoch after the first).
# `reset!` must only be called once the previous producer has fully finished (i.e. after the dataset
# has been fully consumed) — calling it while a producer is still running would race two producers
# on the same buffer.

using Flux
using GraphNeuralNetworks
using Graphs
using Random
using .DataUtils: normalize_graph

export RuntimeDataset, GraphGenerator, ProblemSolver, reset!

"""
    GraphGenerator

Wrapper for a generic graph generator. Implementations must be callable as `gen(rng::AbstractRNG)`.
"""
abstract type GraphGenerator end

"""
    ProblemSolver

Wrapper for a generic problem solver. Must implement a solver (callable as `solver(g)` returning
`(x_0, gap)`) and a heatmap decoder.
"""
abstract type ProblemSolver end

"""
    RuntimeDataset{G<:GraphGenerator, P<:ProblemSolver}

Dataset backed by a seeded `<filepath>/<seed>.csv` (see module docstring above), yielding
`(GNNGraph, name)` pairs from a buffered channel. See `reset!` for replaying the same instances
across multiple passes (e.g. training epochs) without re-materializing them in memory.
"""
struct RuntimeDataset{G<:GraphGenerator, P<:ProblemSolver}
    num_instances::Int
    generator::G
    solver::P
    seed::UInt64
    path::String
    buffer::Channel{Tuple{GNNGraph,String}}
end

"""
    _parse_bitmask(bitmask_str::AbstractString)

Parses a `create_dataset` bitmask column (a string of '0'/'1' chars) into a `(1, n)` Float32 array.
"""
_parse_bitmask(bitmask_str::AbstractString) =
    Float32.(reshape([c == '1' ? 1.0f0 : 0.0f0 for c in bitmask_str], 1, :))

"""
    _spawn_replay_producer!(buffer, num_instances, generator, seed, path)

Spawns a single background task that replays `path` (already-created by `create_dataset`):
`Xoshiro(seed)` regenerates each graph in the same order `create_dataset` used, and the matching CSV
line supplies the already-solved label — no solving happens here. Stops after `num_instances` items.
"""
function _spawn_replay_producer!(buffer::Channel{Tuple{GNNGraph,String}}, num_instances::Int,
                                  generator::GraphGenerator, seed::Integer, path::AbstractString)
    Threads.@spawn begin
        try
            rng = Random.Xoshiro(seed)
            open(path, "r") do io
                readline(io)  # header: "gap,bitmask"
                n = 0
                while n < num_instances
                    if eof(io)
                        @error "Seed dataset $path has fewer than $num_instances lines (stopped at $n) — check num_instances matches the file it was created with."
                        break
                    end
                    _, bitmask_str = split(readline(io), ',')
                    g = generator(rng)
                    x_0 = _parse_bitmask(bitmask_str)
                    gnn_g = GNNGraph(g, ndata=(; x=x_0))

                    put!(buffer, (normalize_graph(gnn_g), "seed_$(seed)_inst_$(n).txt"))
                    n += 1
                end
            end
        catch e
            @error "Seeded dataset producer crashed!" exception=(e, catch_backtrace())
        end
    end
end

"""
    RuntimeDataset(num_instances, generator, solver, seed, filepath; buffer_capacity=32)

Parameters:
    - 'num_instances': The number of instances in the set.
    - 'generator': The generator used for generating instances.
    - 'solver': The solver used for solving instances (only invoked if `<filepath>/<seed>.csv` does not yet exist — see `create_dataset`).
    - 'seed': The RNG seed; also identifies (and, if needed, creates) `<filepath>/<seed>.csv`.
    - 'filepath': The directory `create_dataset` writes/reads `<seed>.csv` in.
    - 'buffer_capacity': The prefetch buffer size.
"""
function RuntimeDataset(num_instances::Int, generator::G, solver::P, seed::Integer, filepath::AbstractString;
                         buffer_capacity::Int=32) where {G<:GraphGenerator, P<:ProblemSolver}
    path = create_dataset(seed, filepath, generator, solver, num_instances)
    buffer = Channel{Tuple{GNNGraph,String}}(buffer_capacity)
    _spawn_replay_producer!(buffer, num_instances, generator, seed, path)
    return RuntimeDataset(num_instances, generator, solver, UInt64(seed), path, buffer)
end

"""
    reset!(dataset::RuntimeDataset)

Respawns the producer task to replay `dataset.path` from the start, supporting another full pass
over the same `num_instances` seeded instances (e.g. the next training epoch) without
re-materializing them in memory and without touching disk again (`create_dataset` is not re-invoked).

Must only be called once the previous producer has fully finished — i.e. after the dataset has
already delivered all `num_instances` items via `getindex`. Calling it while a producer is still
running spawns a second producer racing on the same buffer.
"""
function reset!(dataset::RuntimeDataset)
    _spawn_replay_producer!(dataset.buffer, dataset.num_instances, dataset.generator, dataset.seed, dataset.path)
    return dataset
end

Base.length(d::RuntimeDataset) = d.num_instances

Base.getindex(dataset::RuntimeDataset, ::Int) = take!(dataset.buffer)

function Base.getindex(dataset::RuntimeDataset, inds::AbstractVector{<:Integer})
    return [take!(dataset.buffer) for _ in inds]
end