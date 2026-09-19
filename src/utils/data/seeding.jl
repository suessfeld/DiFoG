# src/utils/data/seeding.jl
#
# Seeded dataset creation. Graphs are not stored: they are regenerated from the seed.
# Only the solution bitmask and its optimality gap are written, one instance per line.

using Random

export create_dataset

"""
    _valid_line_count(path::AbstractString)

Counts complete, well-formed data lines in an existing dataset CSV (header excluded). Stops at the
first line that doesn't split into exactly 2 comma-separated fields — this guards against a
truncated last line left behind by a crash mid-write, so that line is not counted (and is dropped
below before appending, rather than kept as corrupt data).
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

Rewrites `path` to keep only the header plus its first `valid_count` data lines, dropping anything
after (e.g. a crash-truncated trailing line). No-op if the file already has exactly `valid_count`
data lines.
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
    create_dataset(seed::Integer, filepath::AbstractString, generator::GraphGenerator,
                   solver::ProblemSolver, num_instances::Int)

Generates instances from a single `Xoshiro(seed)` stream, solves them, and writes
`<filepath>/<seed>.csv` with one line `gap,bitmask` per instance (line i ↔ instance i).

If `<filepath>/<seed>.csv` already exists:
    - with `≥ num_instances` valid lines: nothing is generated or solved; the path is returned as-is.
    - with `< num_instances` valid lines: the RNG is fast-forwarded past the existing instances
      (by calling `generator(rng)` for each and discarding the graph — cheap, no solving) so the
      newly generated instances continue the exact same stream position they would have had if
      `num_instances` had been requested from the start. Only the missing instances are then solved
      and appended.
    - with a truncated/corrupt trailing line (e.g. a crash mid-write): that line is dropped before
      any of the above, so it is regenerated rather than left as corrupt data.

Generation is sequential, so instance i is fully determined by (seed, i, generator parameters).
Regenerating it means calling `generator(rng)` i times on a fresh `Xoshiro(seed)`; no solving is needed.

Parameters:
    - 'seed': The RNG seed; also the filename.
    - 'filepath': The directory the CSV is written to.
    - 'generator': The graph generator, called as `generator(rng)`.
    - 'solver': The problem solver.
    - 'num_instances': The number of instances the file should contain after this call.
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