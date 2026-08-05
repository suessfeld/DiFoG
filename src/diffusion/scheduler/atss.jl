# src/diffusion/scheduler/atss.jl
using LinearAlgebra: norm

export ATSSScheduler

struct ATSSScheduler{T} <: Scheduler{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
    κ::Function
    κ̇::Function
    atss_p::Float32
end


function _minimize_atss_schedule(T::Int, κ::Function, atss_p::Float32,
                                  fine_grid::Vector{Float32}, x_traj::Vector{Matrix{Float32}})

    lookup_x(t::Float32) = x_traj[argmin(abs.(fine_grid .- t))]

    function cost_and_lambda(t_interior::Vector{Float32})
        t_full = vcat(0f0, t_interior, 1f0)
        total = 0f0
        λ_interior = Vector{Float32}(undef, T - 1)
        for i in 1:T
            t_i, t_im1 = t_full[i + 1], t_full[i]
            Δ_i = norm(vec(lookup_x(t_i)) .- vec(lookup_x(t_im1)))
            α_ti, σ_ti = κ(t_i), 1f0 - κ(t_i)
            ε̃_i = σ_ti^atss_p / max(α_ti, 1f-6)
            λ_i = log(max(α_ti, 1f-6) / max(σ_ti, 1f-6))
            total += ε̃_i * exp(λ_i) * Δ_i
            if i <= T - 1
                λ_interior[i] = λ_i
            end
        end
        return total, λ_interior
    end

    # Candidates = the fine grid's own interior points (excludes t_0=0, t_T=1,
    # which are fixed boundaries) — no separate arbitrary resolution needed,
    # since we already have real trajectory data at every one of these points.
    candidates = fine_grid[2:end-1]

    best_t_interior = fine_grid[round.(Int, range(1, length(fine_grid), length=T+1))][2:T]
    best_cost, best_λ = cost_and_lambda(best_t_interior)

    for j in 1:(T - 1)
        for c in candidates
            trial = copy(best_t_interior)
            trial[j] = c
            sort!(trial)
            c_cost, c_λ = cost_and_lambda(trial)
            if c_cost < best_cost
                best_cost, best_λ, best_t_interior = c_cost, c_λ, trial
            end
        end
    end

    return best_λ
end

"""
    _lambda_inverse(κ::Function, λ::Float32) -> Float32

Inverts λ_t := log(κ(t)/(1-κ(t))) to recover t, via bisection.

DEVIATION FROM PAPER: Algorithm 1 line 12 literally writes t_i ← κ⁻¹(e^{-λ_{t_i}}).
Algebraically, e^{-λ_t} = (1-κ_t)/κ_t ∈ [0,∞), which is NOT in κ's own range
[0,1] — applying κ⁻¹ to it directly is not well-defined (verified: solving
κ_t = (1-κ_t)/κ_t for all t has no solution except one fixed point ≈0.618).
We instead invert the λ(t) relationship itself, matching the reference
implementation's `inverse_lambda` (Xue et al. 2024 code), which is the
well-defined operation Alg. 1 almost certainly intends.
"""
function _lambda_inverse(κ::Function, λ::Float32; lo::Float32=0f0, hi::Float32=1f0, tol::Float32=1f-6)
    f(t) = log(max(κ(t), 1f-6) / max(1f0 - κ(t), 1f-6))
    while hi - lo > tol
        mid = 0.5f0 * (lo + hi)
        if f(mid) < λ
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5f0 * (lo + hi)
end

"""
    _atss_schedule(::Type{VT}, generator, solver, T, κ, κ̇, atss_p, total_steps) where VT

Algorithm 1 (EFLOCO, Li et al. AAAI-26): Adaptive Time-Step Scheduling.
`T` here is exactly the paper's `T`: the number of (adaptive) inference steps —
NOT the model's training-time discretization.

NOTE ON THE "Require" LINE: the paper lists "Target trajectory {x_t}_{t∈[0,1]}"
as a given input, without specifying how it is obtained. Since no such
trajectory pre-exists in this codebase, one is constructed here via a single
rollout (one `generator()` instance, one `solver()` target, one coupled
commit-based interpolation) — this rollout mechanism itself is NOT specified
by the paper; it mirrors `DiscreteFlowMatching.coupled_next_state` to stay
consistent with how this repo already defines the forward corruption process.
The rollout is sampled at `total_steps` resolution (NOT at T) — T only
determines the size of the FINAL returned grid (line 14), while total_steps
is the fine measurement substrate lines 5-9's Δᵢ is computed against.
Everything from line 1 onward is the literal algorithm, operating on top of
this fine trajectory via nearest-point lookup.
"""
function _atss_schedule(::Type{VT}, generator::GraphGenerator, solver::ProblemSolver,
                         T::Int, κ::Function, κ̇::Function, atss_p::Float32,
                         total_steps::Int) where VT

    t_grid = Vector{Float32}(undef, total_steps + 1)   # line 1 (fine substrate, resolution total_steps)
    t_grid[1] = 0f0                          # line 1: t_0 = 0
    t_grid[total_steps + 1] = 1f0            # line 1: t_T = 1  (T here = total_steps, the fine grid's own endpoint)

    for i in 1:(total_steps - 1)             # line 2
        t_grid[i + 1] = Float32(i) / Float32(total_steps)   # line 3: t_i ← i/T
    end                                       # line 4

    # ==== Prep (NOT in Alg. 1 — see docstring): build ONE target trajectory
    #      via ONE rollout, exactly matching "Require: Target trajectory {x_t}"
    #      as a single given input, sampled at total_steps resolution ====
    g   = generator()
    x_1 = solver(g)                          # target solution, shape (1, n)
    n   = size(x_1, 2)

    x_traj = Vector{Matrix{Float32}}(undef, total_steps + 1)   # x_traj[i+1] = x_{t_i}
    x_traj[1] = Float32.(rand(Bool, 1, n))   # x_{t_0}, source ~ Unif{0,1}

    committed = falses(1, n)
    for i in 1:total_steps
        t_i   = t_grid[i + 1]
        t_im1 = t_grid[i]

        κ_i, κ_im1 = κ(t_i), κ(t_im1)
        p_commit = clamp((κ_i - κ_im1) / max(1f0 - κ_im1, 1f-6), 0f0, 1f0)

        fresh_noise  = Float32.(rand(Bool, 1, n))
        newly_commit = (.!committed) .& (rand(Float32, 1, n) .< p_commit)
        committed  .|= newly_commit

        x_traj[i + 1] = ifelse.(committed, x_1, fresh_noise)
    end

    # ==== Alg. 1, lines 5-9 (literal, evaluated against the fine substrate) ====
    ε̃ = Vector{Float32}(undef, total_steps)   # ε̃[i] = ε̃_{t_i}
    λ  = Vector{Float32}(undef, total_steps)   # λ[i]  = λ_{t_i}, Eq. 12
    L  = Vector{Float32}(undef, total_steps)   # L[i]  = L_i, Eq. 14
    Δ = Vector{Float32}(undef, total_steps)

    for i in 1:total_steps   # line 5
        t_i = t_grid[i + 1]

        # line 6: State change Δᵢ ← ‖x_{t_i} - x_{t_{i-1}}‖   (Eq. 13)
        Δ[i] = norm(vec(x_traj[i + 1]) .- vec(x_traj[i]))

        α_ti = κ(t_i)                      # α_t := κ(t)   (target-side weight)
        σ_ti = 1f0 - κ(t_i)                # σ_t := 1-κ(t) (source-side weight)

        # line 7: Path weight ε̃_{t_i} ← σ_{t_i}^p / α_{t_i}   (Eq. 12's σ^p/α term)
        ε̃[i] = σ_ti^atss_p / max(α_ti, 1f-6)

        # Eq. 12: λ_{t_i} := log(α_{t_i} / σ_{t_i})  — needed for line 8's cost
        λ[i] = log(max(α_ti, 1f-6) / max(σ_ti, 1f-6))

        # line 8: Cost L_i ← ε̃_{t_i} · e^{λ_{t_i}} · Δᵢ   (Eq. 14)
        L[i] = ε̃[i] * exp(λ[i]) * Δ[i]
    end   # line 9

    # line 10: {λ_{t_1},...,λ_{t_{T-1}}} ← minimizing Σ_{i=1}^T L_i
    # (search runs over T candidate interior points, T = inference_steps,
    #  NOT total_steps — Δᵢ during the search is looked up against the
    #  fine t_grid/x_traj built above)
    λ_interior = _minimize_atss_schedule(T, κ, atss_p, t_grid, x_traj)

    # ==== Alg. 1, lines 11-13 (literal) ====
    t_interior = Vector{Float32}(undef, T - 1)
    for i in 1:(T - 1)                                    # line 11
        t_interior[i] = _lambda_inverse(κ, λ_interior[i])  # line 12: t_i ← κ⁻¹(...) [see _lambda_inverse docstring for deviation]
    end                                                    # line 13

    return vcat(0f0, t_interior, 1f0)   # line 14: {t_0, t_1, ..., t_T}
end

function ATSSScheduler(::Type{VT}, generator::GraphGenerator, solver::ProblemSolver,
                        total_steps::Int, inference_steps::Int,
                        κ::Function, κ̇::Function;
                        atss_p::Float32=1.0f0) where VT

    if !(VT === Int || VT === Float32)
        error("ATSSScheduler only supports Scheduler{Int} or Scheduler{Float32}; got $VT")
    end

    steps_raw = _atss_schedule(VT, generator, solver, inference_steps, κ, κ̇, atss_p, total_steps)

    steps = VT === Int ? round.(Int, clamp.(steps_raw .* total_steps, 1, total_steps)) : steps_raw

    return ATSSScheduler{VT}(total_steps, inference_steps, steps, κ, κ̇, atss_p)
end

# ==== NEW — nothing above this line changes ====
# ---- Convenience default (mirrors LinearScheduler/CosineScheduler's bare overload) ----
ATSSScheduler(generator::GraphGenerator, solver::ProblemSolver,
              total_steps::Int, inference_steps::Int,
              κ::Function, κ̇::Function; atss_p::Float32=1.0f0) =
    ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; atss_p=atss_p)
# ==== end NEW ====

ATSSScheduler(::Type{VT}, s::ATSSScheduler{VT}) where {VT} = s   # identity, avoid re-deriving

function ATSSScheduler(::Type{Int}, s::ATSSScheduler{Float32})
    steps = round.(Int, clamp.(s.steps .* s.total_steps, 1, s.total_steps))
    return ATSSScheduler{Int}(s.total_steps, s.inference_steps, steps, s.κ, s.κ̇, s.atss_p)
end

function ATSSScheduler(::Type{Float32}, s::ATSSScheduler{Int})
    steps = Float32.(s.steps) ./ Float32(s.total_steps)
    return ATSSScheduler{Float32}(s.total_steps, s.inference_steps, steps, s.κ, s.κ̇, s.atss_p)
end

function ATSSScheduler(total_steps::Int, inference_steps::Int, steps::AbstractVector{T},
                        κ::Function, κ̇::Function, atss_p::Float32) where T
    return ATSSScheduler{T}(total_steps, inference_steps, Array(steps), κ, κ̇, atss_p)
end

Base.convert(::Type{Scheduler{VT}}, s::ATSSScheduler) where {VT} = ATSSScheduler(VT, s)
Base.convert(::Type{ATSSScheduler{VT}}, s::ATSSScheduler) where {VT} = ATSSScheduler(VT, s)