# src/utils/data/seeding.jl
#
# Seeded dataset creation. Graphs are not stored: they are regenerated from the seed.
# Only the solution bitmask and its optimality gap are written, one instance per line.

using Random

export create_dataset

"""
    _valid_line_count(path::AbstractString)

Counts complete, well-formed data lines in an existing dataset CSV, excluding the header. Stops at the first malformed line to guard against truncated data from crashes.

Parameters:
    - 'path': The file path to the CSV dataset.
"""
function _valid_line_count(path::AbstractString)
    n = 0
    open(path, "r") do io
        readline(io)  # header
        for line in eachline(io)
            length(split(line, ',')) == 2 || break
            n += 1
        end
    end
    return n
end

"""
    _truncate_to_valid!(path::AbstractString, valid_count::Int)

Rewrites the dataset CSV to retain only the header and the specified number of valid data lines, discarding any subsequent corrupted lines.

Parameters:
    - 'path': The file path to the CSV dataset.
    - 'valid_count': The number of valid data lines to retain.
"""
function _truncate_to_valid!(path::AbstractString, valid_count::Int)
    lines = readlines(path)
    kept = lines[1:min(1 + valid_count, length(lines))]  # header + first valid_count data lines
    length(kept) == length(lines) && return
    open(path, "w") do io
        for line in kept
            println(io, line)
        end
    end
end

"""
    create_dataset(seed::Integer, filepath::AbstractString, generator::GraphGenerator, solver::ProblemSolver, num_instances::Int)

Generates and solves graph instances from a seeded random stream, appending missing instances to a CSV file if it already exists. Ensures the RNG stream is fast-forwarded to maintain deterministic generation.

Parameters:
    - 'seed': The RNG seed used for deterministic generation and as the filename.
    - 'filepath': The directory where the CSV is written.
    - 'generator': The graph generator used to instantiate graphs.
    - 'solver': The problem solver used to compute the exact solutions.
    - 'num_instances': The target total number of instances the file should contain.
"""
function create_dataset(seed::Integer, filepath::AbstractString, generator::GraphGenerator,
                        solver::ProblemSolver, num_instances::Int)
    mkpath(filepath)
    path = joinpath(filepath, "$(seed).csv")
    rng = Random.Xoshiro(seed)

    existing = 0
    if isfile(path)
        existing = _valid_line_count(path)
        _truncate_to_valid!(path, existing)

        if existing >= num_instances
            @info "Dataset $path already has $existing ≥ $num_instances instances — reusing it."
            return path
        end

        @info "Dataset $path has $existing instances — generating $(num_instances - existing) more to reach $num_instances."
        for _ in 1:existing
            generator(rng)  # discard: fast-forward the RNG stream past already-written instances
        end
    end

    open(path, existing > 0 ? "a" : "w") do io
        existing == 0 && println(io, "gap,bitmask")
        for _ in (existing + 1):num_instances
            g = generator(rng)
            x_0, gap = solver(g)
            println(io, gap, ',', join(Int.(vec(x_0) .> 0.5f0)))
            flush(io)
        end
    end
    return path
end