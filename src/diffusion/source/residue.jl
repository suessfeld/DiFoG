# src/diffusion/source/residue.jl
#
# Residue source following DISCO (Zhao et al., 2025): instead of pure noise, generation starts from a cheap
# degraded solution X_d plus noise, x₀ = signal(X_d) + σ ε, so the model only has to learn the residue between
# X_d and a good solution. Two kinds of degraded solution are supported:
#   - :heuristic — the problem-specific degraded solution `residue(solver, g)` of the problem file
#                  (e.g. the degree greedy for the MISP); an instance-specific extension of DISCO,
#   - :random    — a uniformly random node subset (each node with probability 1/2), which is what DISCO
#                  uses for the MIS; it carries no information about the instance.

using Random
using Functors
using Flux
using GraphNeuralNetworks

export ResidueSource

"""
    ResidueSource{P<:ProblemSolver} <: SourceDistribution

Source around a degraded solution: x₀ = signal(X_d) + σ ε with ε ~ N(0, I).

Fields:
    - 'solver': The problem solver providing `residue(solver, g)` (used by `kind = :heuristic`).
    - 'kind': `:heuristic` (problem-specific `residue`) or `:random` (uniform random node subset, as DISCO for the MIS).
    - 'σ': Standard deviation of the noise added to the degraded solution (≥ 0; 0 starts exactly at X_d).
"""
struct ResidueSource{P<:ProblemSolver} <: SourceDistribution
    solver::P
    kind::Symbol
    σ::Float32

    function ResidueSource(solver::P; kind::Symbol=:heuristic, σ::Real=1f0) where {P<:ProblemSolver}
        kind in (:heuristic, :random) ||
            throw(ArgumentError("ResidueSource: kind must be :heuristic or :random, got :$kind"))
        σ >= 0 || throw(ArgumentError("ResidueSource: σ must be ≥ 0, got $σ"))
        return new{P}(solver, kind, Float32(σ))
    end
end

Functors.@leaf ResidueSource

"""
    degraded_solution(source::ResidueSource, g::GNNGraph, shape::Tuple)

Returns the degraded solution X_d in label space {0, 1} with size `shape`, on the device of `g`. For
`:heuristic`, `residue(solver, ·)` is evaluated per graph of the batch on the CPU and the results are
concatenated in the node order of the batch; for `:random`, every node is drawn independently.

Parameters:
    - 'source': The residue source.
    - 'g': The (batched) graph.
    - 'shape': The dimensions of the state (node-state width × number of nodes).
"""
function degraded_solution(source::ResidueSource, g::GNNGraph, shape::Tuple)
    anchor = _device_anchor(g)
    if source.kind === :random
        return Float32.(rand!(similar(anchor, Float32, shape)) .> 0.5f0)
    end

    g_cpu = g |> Flux.cpu
    graphs = g_cpu.num_graphs == 1 ? [g_cpu] : Flux.unbatch(g_cpu)
    x_d = reduce(vcat, [Float32.(vec(residue(source.solver, gi))) for gi in graphs])
    @assert length(x_d) == shape[2] "residue: got $(length(x_d)) node values for $(shape[2]) nodes"
    return copyto!(similar(anchor, Float32, shape), repeat(reshape(x_d, 1, :), shape[1], 1))
end

"""
    sample_source(source::ResidueSource, g::GNNGraph, shape::Tuple)

Samples x₀ = signal(X_d) + σ ε on the device of `g`.
"""
function sample_source(source::ResidueSource, g::GNNGraph, shape::Tuple)
    x_d = _to_signal(degraded_solution(source, g, shape))
    iszero(source.σ) && return x_d
    return x_d .+ source.σ .* randn!(similar(x_d))
end

source_label(source::ResidueSource) =
    "Residue($(source.kind), sigma=$(source.σ), $(nameof(typeof(source.solver))))"
