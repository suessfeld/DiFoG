# src/diffusion/source.jl
#
# Defines the abstract SourceDistribution type: the distribution p₀ a generative method starts from.
# A method only calls `sample_source` whenever it needs a source sample; which distribution is drawn from
# (pure noise, or noise around a degraded solution) is decided by the concrete source at runtime.
# Implementations live in src/diffusion/source/ (analogous to the schedulers in src/diffusion/scheduler/).

export SourceDistribution, sample_source, source_label

"""
    SourceDistribution

An abstract base type for source distributions p₀ of continuous generative methods. Implementations must
provide `sample_source` and `source_label`.
"""
abstract type SourceDistribution end

"""
    sample_source(source::SourceDistribution, g::GNNGraph, shape::Tuple)

Draws one source state of size `shape` (node-state width × number of nodes of the batch) for the graphs in
`g`, on the device of `g`. The state is in signal space (labels 0/1 correspond to −1/+1).

Parameters:
    - 'source': The source distribution.
    - 'g': The (batched) graph; determines the device and, for problem-specific sources, the instances.
    - 'shape': The dimensions of the state.
"""
function sample_source end

"""
    source_label(source::SourceDistribution)

Short human-readable description of the source, used in reports.
"""
function source_label end
