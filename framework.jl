# ==========================================================================
# Unified discrete diffusion / discrete flow matching / Gaussian diffusion
# for node-selection problems on graphs (e.g. Maximum Independent Set).
#
# Time convention: t = 0 is the source distribution, t = 1 is clean data.
# Schedulers κ : [0,1] → [0,1] satisfy κ(0) = 0, κ(1) = 1 and act as:
#   * fraction of data bits kept  (Bernoulli D3PM, DFM mixture path)
#   * cumulative signal αbar(t)   (Gaussian DDPM/DDIM)
# ==========================================================================

# ----- Generative frameworks -------------------------------------------

"""
    GenerativeFramework

Abstract type for generative frameworks.
"""
abstract type GenerativeFramework end

"""
    DiscreteDiffusion

DIFUSCO-style discrete (Bernoulli D3PM) diffusion.
#GR TODO Bitte auch hier im Source für die zentralen nachimplementierten Aspekte die
entsprechende Literaturreferenz angeben, wie hier zum DIFUSCO Paper
"""
struct DiscreteDiffusion <: GenerativeFramework end

"""
    DiscreteFlowMatching

Discrete flow matching (CTMC mixture path with Euler integration).
"""
struct DiscreteFlowMatching <: GenerativeFramework end

"""
    GaussianDiffusion

DIFUSCO-style continuous (Gaussian) diffusion with DDIM sampling.
"""
struct GaussianDiffusion <: GenerativeFramework end

# ----- Schedulers κ : [0,1] → [0,1] ------------------------------------
abstract type Scheduler end
struct LinearScheduler <: Scheduler end
struct CosineScheduler <: Scheduler end
struct PolyScheduler   <: Scheduler ; k::Float64 ; end

"""
    κ(sched, t)

Fraction of signal preserved at time `t` under scheduler `sched`. Equals
the keep probability for discrete frameworks and ``\\bar{\\alpha}(t)`` for
Gaussian diffusion.
"""
κ( ::LinearScheduler, t) = t
κ( ::CosineScheduler, t) = sin(π*t/2)^2
κ(s::PolyScheduler,   t) = t^s.k

"""
    dκ(sched, t)

Time derivative of [`κ`](@ref). Required by the DFM Euler step.
"""
dκ( ::LinearScheduler, t) = 1.0
dκ( ::CosineScheduler, t) = (π/2)*sin(π*t)
dκ(s::PolyScheduler,   t) = s.k*t^(s.k-1)

# ----- Inference timestep grids ----------------------------------------
abstract type InferenceSchedule end
struct LinearInference <: InferenceSchedule end
struct CosineInference <: InferenceSchedule end

"""
    timesteps(inf_sched, N)

Return `N+1` ascending timesteps in `[0, 1]` selected according to
`inf_sched`. The first entry is `0` (source) and the last is `1` (data).
"""
timesteps(::LinearInference, N) = collect(range(0.0, 1.0; length = N+1))
timesteps(::CosineInference, N) =
    [sin(π*s/2)^2 for s in range(0.0, 1.0; length = N+1)]

# ----- Source distributions --------------------------------------------
"""
    sample_source(framework, n; kwargs...)

Sample an initial state at `t = 0` for a graph with `n` nodes. Bernoulli
with rate `π₀` for discrete frameworks; standard normal for Gaussian.
"""
sample_source(::DiscreteDiffusion,    n; π₀ = 0.5) = rand(n) .< π₀
sample_source(::DiscreteFlowMatching, n; π₀ = 0.5) = rand(n) .< π₀
sample_source(::GaussianDiffusion,    n; kwargs...) = randn(n)

# ----- Forward (corruption) process ------------------------------------
"""
    corrupt(framework, sched, x₁, t; kwargs...)

Sample `x_t` given clean data `x₁`, scheduler `sched`, and time `t`.

For discrete frameworks each bit of `x₁` is kept with probability
``κ(t)`` and otherwise resampled from the source. For Gaussian diffusion
``x_t = \\sqrt{κ(t)}\\,x_1 + \\sqrt{1 - κ(t)}\\,ε`` with
``ε \\sim \\mathcal{N}(0, I)``, matching the DDPM ``\\bar{\\alpha}``
parametrisation.
"""
function corrupt(fw::Union{DiscreteDiffusion,DiscreteFlowMatching},
                 sched::Scheduler, x₁::BitVector, t; π₀ = 0.5)
    keep = rand(length(x₁)) .< κ(sched, t)
    return ifelse.(keep, x₁, sample_source(fw, length(x₁); π₀))
end

function corrupt(::GaussianDiffusion, sched::Scheduler,
                 x₁::AbstractVector{<:Real}, t; kwargs...)
    αbar = κ(sched, t)
    ε    = randn(length(x₁))
    return @. √αbar * x₁ + √(1 - αbar) * ε
end

# ----- Denoiser (GNN) --------------------------------------------------
struct Denoiser end   # placeholder for an anisotropic GNN with time embedding

"""
    predict_target(model, framework, G, x_t, t)

Run the denoiser on graph `G` and noisy state `x_t` at time `t`.

Returns per-node probabilities ``\\hat p_1^i = P(x_1^i = 1 \\mid x_t, G)``
for discrete frameworks, or the predicted clean state ``\\hat x_1 \\in ℝ^n``
for Gaussian diffusion (`x₁`-prediction parametrisation).
"""
function predict_target(model::Denoiser,
                        ::Union{DiscreteDiffusion,DiscreteFlowMatching},
                        G, x_t, t)::Vector{Float64}
    # logits = forward(model, G, Float32.(x_t), t); return σ.(logits)
end

function predict_target(model::Denoiser, ::GaussianDiffusion,
                        G, x_t, t)::Vector{Float64}
    # return forward(model, G, x_t, t)   # x₁-prediction parametrisation
end

# ----- Per-framework training loss -------------------------------------
"""
    loss(framework, x₁, prediction)

Per-framework supervised loss between ground truth `x₁` and `prediction`:
binary cross-entropy for discrete frameworks, mean squared error for
Gaussian diffusion.
"""
function loss(::Union{DiscreteDiffusion,DiscreteFlowMatching},
              x₁::BitVector, p̂₁::AbstractVector{<:Real})
    return -mean(@. x₁*log(p̂₁ + 1e-9) + (1 - x₁)*log(1 - p̂₁ + 1e-9))
end

function loss(::GaussianDiffusion, x₁::AbstractVector{<:Real},
              x̂₁::AbstractVector{<:Real})
    return mean(abs2, x₁ .- x̂₁)
end

# ----- Generic training step -------------------------------------------
"""
    train_step!(model, G, x₁, framework, sched; kwargs...)

Run a single gradient step: sample a time `t`, corrupt `x₁` to `x_t`,
predict the target, compute the framework-specific loss and update
`model`. Replace `rand()` with logit-normal sampling for SD3-style
training.
"""
function train_step!(model, G, x₁, framework::GenerativeFramework,
                     sched::Scheduler; kwargs...)
    t   = rand()
    x_t = corrupt(framework, sched, x₁, t; kwargs...)
    p̂   = predict_target(model, framework, G, x_t, t)
    L   = loss(framework, x₁, p̂)
    backprop!(model, L)
    return L
end

# ----- Per-framework denoising step ------------------------------------
"""
    denoise_step(framework, sched, x_t, t, t′, prediction)

Take one reverse-process step from `t` to `t′ > t` (`t = 0` is source,
`t = 1` is data).

# Implementation
* `DiscreteDiffusion` — sample per node from the mixture-path posterior
  ``q(x_{t′} \\mid x_t, \\hat x_1)``: keep `x_t` with probability
  ``α = (1 - κ(t′)) / (1 - κ(t))``, else resample from ``\\hat p_1``.
* `DiscreteFlowMatching` — Euler step on the per-node CTMC with flip
  rate ``\\dot κ(t) / (1 - κ(t)) \\cdot P(x_1 ≠ x_t)``.
* `GaussianDiffusion` — deterministic DDIM step in ``\\bar{\\alpha}``
  parametrisation: recover ``\\hat ε`` from ``\\hat x_1`` and renoise to
  level `t′`.
"""
function denoise_step(::DiscreteDiffusion, sched::Scheduler,
                      x_t::BitVector, t, t′, p̂₁::AbstractVector{<:Real})
    α = (1 - κ(sched, t′)) / (1 - κ(sched, t) + 1e-9)
    return BitVector(rand() < α ? x_t[i] : (rand() < p̂₁[i])
                     for i in eachindex(x_t))
end

function denoise_step(::DiscreteFlowMatching, sched::Scheduler,
                      x_t::BitVector, t, t′, p̂₁::AbstractVector{<:Real})
    Δt          = t′ - t
    rate_factor = dκ(sched, t) / (1 - κ(sched, t) + 1e-9)
    out         = BitVector(undef, length(x_t))
    for i in eachindex(x_t)
        p_other = x_t[i] ? (1 - p̂₁[i]) : p̂₁[i]
        flip    = clamp(Δt * rate_factor * p_other, 0.0, 1.0)
        out[i]  = rand() < flip ? !x_t[i] : x_t[i]
    end
    return out
end

function denoise_step(::GaussianDiffusion, sched::Scheduler,
                      x_t::AbstractVector{<:Real}, t, t′,
                      x̂₁::AbstractVector{<:Real})
    αbar_t  = κ(sched, t)
    αbar_t′ = κ(sched, t′)
    ε̂ = (x_t .- √αbar_t .* x̂₁) ./ √(1 - αbar_t + 1e-9)
    return @. √αbar_t′ * x̂₁ + √(1 - αbar_t′) * ε̂
end

# ----- Generic sampler -------------------------------------------------
"""
    sample(model, G, framework, sched, inf_sched; N = 50, kwargs...)

Run the reverse process from `t = 0` to `t = 1` over `N` steps using the
chosen framework, scheduler and inference grid. Returns the terminal
state at `t = 1`.
"""
function sample(model, G, framework::GenerativeFramework,
                sched::Scheduler, inf_sched::InferenceSchedule;
                N = 50, kwargs...)
    x_t = sample_source(framework, nv(G); kwargs...)
    ts  = timesteps(inf_sched, N)
    for k in 1:N
        p̂   = predict_target(model, framework, G, x_t, ts[k])
        x_t = denoise_step(framework, sched, x_t, ts[k], ts[k+1], p̂)
    end
    return x_t
end

# ----- Final score and feasibility projection --------------------------
"""
    final_score(framework, x_final, model, G)

Convert the terminal state of the reverse process into a per-node score
in ``[0, 1]`` suitable for [`project_mis`](@ref). For Gaussian diffusion
the continuous output is squashed via the logistic; for discrete
frameworks the model is queried once at `t = 1` for clean probabilities.
"""
function final_score(fw::Union{DiscreteDiffusion,DiscreteFlowMatching},
                     x_final, model, G)
    return predict_target(model, fw, G, x_final, 1.0)
end

function final_score(::GaussianDiffusion, x_final::AbstractVector{<:Real},
                     model, G)
    return @. 1 / (1 + exp(-x_final))
end

"""
    project_mis(G, score)

Greedily decode a per-node `score` vector into a feasible independent set
on `G`: process nodes in descending score order, accept a node if no
neighbour has been selected yet, and exclude its neighbours afterwards.
"""
function project_mis(G, score::AbstractVector{<:Real})
    chosen, excluded = falses(nv(G)), falses(nv(G))
    for v in sortperm(score; rev = true)
        excluded[v] && continue
        chosen[v] = true
        for u in neighbors(G, v); excluded[u] = true; end
    end
    return chosen
end

# ----- End-to-end MIS solver -------------------------------------------
"""
    solve_mis(model, G, framework, sched, inf_sched; N = 50, n_samples = 8)

Sample `n_samples` independent denoising trajectories, project each to a
feasible independent set on `G` and return the largest one found.
"""
function solve_mis(model, G, framework::GenerativeFramework,
                   sched::Scheduler, inf_sched::InferenceSchedule;
                   N = 50, n_samples = 8)
    best = falses(nv(G))
    for _ in 1:n_samples
        x_end = sample(model, G, framework, sched, inf_sched; N)
        score = final_score(framework, x_end, model, G)
        sol   = project_mis(G, score)
        sum(sol) > sum(best) && (best = sol)
    end
    return best
end