# src/diffusion/source/gaussian.jl
#
# Pure Gaussian source x₀ ~ N(0, σ²I), the standard source of Rectified Flow (Liu et al., 2022).

using Random
using Functors

export GaussianSource

"""
    GaussianSource <: SourceDistribution

Pure Gaussian noise without any information about the instance.

Fields:
    - 'σ': Standard deviation of the noise (> 0).
"""
struct GaussianSource <: SourceDistribution
    σ::Float32

    function GaussianSource(σ::Real=1f0)
        σ > 0 || throw(ArgumentError("GaussianSource: σ must be > 0, got $σ"))
        return new(Float32(σ))
    end
end

Functors.@leaf GaussianSource

"""
    sample_source(source::GaussianSource, g::GNNGraph, shape::Tuple)

Samples σ·ε with ε ~ N(0, I) directly on the device of `g`.
"""
sample_source(source::GaussianSource, g::GNNGraph, shape::Tuple) =
    source.σ .* randn!(similar(_device_anchor(g), Float32, shape))

source_label(source::GaussianSource) = "Gaussian(sigma=$(source.σ))"
