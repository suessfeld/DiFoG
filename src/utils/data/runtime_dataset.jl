# src/utils/data/runtime_dataset.jl
#
# Implements a seeded, asynchronous dataset generator that yields graph instances and their exact
# solver solutions.
#
# Ensures `<filepath>/<seed>[_<tag>].csv` exists via `create_dataset` (creating it if needed, reusing it
# otherwise), then replays it: regenerates each graph from a fresh `Xoshiro(seed)` (matching
# `create_dataset`'s generation order) and reads its already-solved label and gap from the CSV instead of
# solving it again. This is necessarily sequential — the RNG stream must be consumed in order.
#
# The producer task is finite: it stops after exactly `num_instances` items, so a `RuntimeDataset`
# supports ONE end-to-end pass per producer. `reset!` finishes (or drains) the previous producer and
# spawns a new one for the next pass (called by `train_diffusion_model!` once per epoch after the first).
# If the producer fails (e.g. the CSV is too short), it closes the buffer with the error so consumers
# fail loudly instead of blocking forever (R16).

using Flux
using GraphNeuralNetworks
using Graphs
using Random
using .DataUtils: normalize_graph

export RuntimeDataset, GraphGenerator, ProblemSolver, reset!, label_symmetric, dataset_tag, base_solver

"""
    GraphGenerator

An abstract base type representing a generic graph generator. Implementations must be callable as `gen(rng::AbstractRNG)`.
"""
abstract type GraphGenerator end

"""
    ProblemSolver

An abstract base type representing a generic problem solver. Implementations must provide a solver callable, `decode_heatmap`, `evaluate`, and `degree_baseline_heatmap`.
"""
abstract type ProblemSolver end

"""
    label_symmetric(solver::ProblemSolver)

Indicates whether the problem's labels are only defined up to their complement, enabling flip-invariant training loss.

Parameters:
    - 'solver': The ProblemSolver instance (defaults to false for generic solvers).
"""
label_symmetric(::ProblemSolver) = false

"""
    dataset_tag(solver::ProblemSolver)

Suffix that distinguishes the dataset files of different labelling solvers for the same seed, i.e. for the
same graphs: `<seed>.csv` for an empty tag (the default, used by the exact solvers) and `<seed>_<tag>.csv`
otherwise.

Parameters:
    - 'solver': The ProblemSolver instance (defaults to "" for generic solvers).
"""
dataset_tag(::ProblemSolver) = ""

"""
    canonicalize_label!(solver::ProblemSolver, x::AbstractMatrix, g)

Brings a label into the canonical form of the problem, if its solutions have several equivalent encodings
(e.g. a cut and its complement). The default leaves the label unchanged.

Parameters:
    - 'solver': The ProblemSolver instance.
    - 'x': The label as a (1, n) matrix, modified in place and returned.
    - 'g': The graph of the instance.
"""
canonicalize_label!(::ProblemSolver, x::AbstractMatrix, g) = x

"""
    base_solver(solver::ProblemSolver)

The solver that defines the problem, i.e. the solver itself, or the wrapped solver of a solver that only
changes how labels are computed (e.g. `LNSSolver`).
"""
base_solver(solver::ProblemSolver) = solver

"""
    RuntimeDataset{G<:GraphGenerator, P<:ProblemSolver}

A dataset backed by a seeded CSV file, yielding graph instances and their exact solver solutions via a buffered channel.

Fields:
    - 'num_instances': The total number of instances in the dataset.
    - 'generator': The graph generator used to recreate instances.
    - 'solver': The exact solver used for label generation.
    - 'seed': The random seed used for generation and file identification.
    - 'path': The file path to the underlying CSV dataset.
    - 'buffer': An asynchronous channel buffering generated graph-label pairs.
    - 'producer': A reference to the background task filling the buffer.
"""
struct RuntimeDataset{G<:GraphGenerator, P<:ProblemSolver}
    num_instances::Int
    generator::G
    solver::P
    seed::UInt64
    path::String
    buffer::Channel{Tuple{GNNGraph,String}}
    producer::Base.RefValue{Task}
end

"""
    _parse_bitmask(bitmask_str::AbstractString)

Parses a string of binary characters ('0'/'1') into a Float32 array mask.

Parameters:
    - 'bitmask_str': The string representation of the bitmask.
"""
_parse_bitmask(bitmask_str::AbstractString) =
    Float32.(reshape([c == '1' ? 1.0f0 : 0.0f0 for c in bitmask_str], 1, :))

"""
    _parse_gap(gap_str::AbstractString)

Parses the optimality gap string into a Float64, defaulting to NaN for unparsable values.

Parameters:
    - 'gap_str': The string representation of the optimality gap.
"""
_parse_gap(gap_str::AbstractString) = something(tryparse(Float64, strip(gap_str)), NaN)

"""
    _spawn_replay_producer!(buffer::Channel{Tuple{GNNGraph,String}}, num_instances::Int, generator::GraphGenerator, seed::Integer, path::AbstractString)

Spawns a background task that sequentially replays the seeded dataset from the CSV file into the buffer.

Parameters:
    - 'buffer': The asynchronous channel to put generated items into.
    - 'num_instances': The number of items to read and generate.
    - 'generator': The graph generator.
    - 'seed': The random seed used to ensure deterministic graph recreation.
    - 'path': The path to the CSV file containing labels and gaps.
"""
function _spawn_replay_producer!(buffer::Channel{Tuple{GNNGraph,String}}, num_instances::Int,
                                 generator::GraphGenerator, seed::Integer, path::AbstractString)
    return Threads.@spawn begin
        try
            rng = Random.Xoshiro(seed)
            open(path, "r") do io
                readline(io)  # header: "gap,bitmask"
                for n in 0:(num_instances - 1)
                    eof(io) && error("Seed dataset $path has fewer than $num_instances lines (stopped at $n) — check num_instances matches the file it was created with.")
                    gap_str, bitmask_str = split(readline(io), ',')
                    g = generator(rng)
                    x_0 = _parse_bitmask(bitmask_str)
                    gnn_g = GNNGraph(g, ndata=(; x=x_0), gdata=(; gap=Float32[_parse_gap(gap_str)]))

                    put!(buffer, (normalize_graph(gnn_g), "seed_$(seed)_inst_$(n).txt"))
                end
            end
        catch e
            @error "Seeded dataset producer crashed — closing the buffer so consumers fail instead of waiting." exception=(e, catch_backtrace())
            close(buffer, e isa Exception ? e : ErrorException(string(e)))
            rethrow()
        end
    end
end

"""
    RuntimeDataset(num_instances::Int, generator::G, solver::P, seed::Integer, filepath::AbstractString; buffer_capacity::Int=32) where {G<:GraphGenerator, P<:ProblemSolver}

Constructs a new RuntimeDataset, initializing the CSV backing file if necessary and spawning the initial producer task.

Parameters:
    - 'num_instances': The number of instances in the set.
    - 'generator': The generator used for generating instances.
    - 'solver': The solver (only invoked if the CSV does not yet exist).
    - 'seed': The RNG seed identifying the dataset.
    - 'filepath': The directory path for the dataset CSV file.
    - 'buffer_capacity': The prefetch buffer size for the channel (default is 32).
"""
function RuntimeDataset(num_instances::Int, generator::G, solver::P, seed::Integer, filepath::AbstractString;
                         buffer_capacity::Int=32) where {G<:GraphGenerator, P<:ProblemSolver}
    path = create_dataset(seed, filepath, generator, solver, num_instances)
    buffer = Channel{Tuple{GNNGraph,String}}(buffer_capacity)
    task = _spawn_replay_producer!(buffer, num_instances, generator, seed, path)
    return RuntimeDataset(num_instances, generator, solver, UInt64(seed), path, buffer, Ref(task))
end

"""
    reset!(dataset::RuntimeDataset)

Resets the dataset for a new pass by draining leftover items in the buffer, awaiting the previous task, and spawning a new replay producer.

Parameters:
    - 'dataset': The RuntimeDataset instance to reset.
"""
function reset!(dataset::RuntimeDataset)
    task = dataset.producer[]
    drained = 0
    while !istaskdone(task)
        if isready(dataset.buffer)
            take!(dataset.buffer); drained += 1
        else
            sleep(0.01)
        end
    end
    wait(task)                                   # rethrows a failed producer
    while isready(dataset.buffer)                # leftovers of an aborted pass
        take!(dataset.buffer); drained += 1
    end
    drained > 0 && @warn "reset!: discarded $drained unconsumed instances of the previous (aborted) pass."
    dataset.producer[] = _spawn_replay_producer!(dataset.buffer, dataset.num_instances,
                                                 dataset.generator, dataset.seed, dataset.path)
    return dataset
end

Base.length(d::RuntimeDataset) = d.num_instances

Base.getindex(dataset::RuntimeDataset, ::Int) = take!(dataset.buffer)

function Base.getindex(dataset::RuntimeDataset, inds::AbstractVector{<:Integer})
    return [take!(dataset.buffer) for _ in inds]
end