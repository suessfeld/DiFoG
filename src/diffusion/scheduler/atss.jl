# src/diffusion/scheduler/atss.jl
#
# Implements the Adaptive Time-Step Scheduling (ATSS) strategy from the EFLOCO paper (Eq. 12–14,
# Algorithm 1). Used for INFERENCE spacing only; training time is sampled uniformly (the training
# density ρ(t) of Eq. 15 is intentionally not used, see project decisions).
#
# Algorithm 1 only needs the signal share κ(t) of the method's path (for the weight of Eq. 12–14) and one
# trajectory along that path (for Δᵢ, Eq. 13). It is applied to the path of each method (`path`):
#   - :flow_matching  — mixture path of DFM, κ as passed (EFLOCO), trajectory: each node switches from the
#                       uniform source to the target at a random time with P(switched by t) = κ(t).
#   - :diffusion      — the Bernoulli forward process of DDPM keeps x₀ with share ᾱ and is uniform noise
#                       otherwise, i.e. it is a mixture path with κ(τ) = ᾱ at diffusion time T(1 − τ); same
#                       switching trajectory with that κ.
#   - :rectified_flow — straight Gaussian path x_τ = τ·x₁ + (1 − τ)·x₀ with x₁ ∈ {−1, +1}, x₀ ~ N(0, I),
#                       κ(τ) = τ; the trajectory is that straight line (Δᵢ is then proportional to the step length).

using LinearAlgebra: norm
using Functors

export ATSSScheduler

"""
    ATSSScheduler{T} <: TimeSchedule{T}

Implements the Adaptive Time-Step Scheduling (ATSS) strategy (Algorithm 1 of the EFLOCO paper).

Fields:
    - 'total_steps': Resolution of the fine grid used to evaluate the trajectory.
    - 'inference_steps': The number of adaptive inference steps.
    - 'steps': The adaptive time-step sequence.
    - 'κ': The interpolation function.
    - 'κ̇': The derivative of the interpolation function.
    - 'atss_p': The exponent p of the path weight.
    - 'path': The method path the schedule was fitted on (:flow_matching, :diffusion or :rectified_flow).
"""
struct ATSSScheduler{T} <: TimeSchedule{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
    κ::Function
    κ̇::Function
    atss_p::Float32
    path::Symbol
end

const ATSS_PATHS = (:flow_matching, :diffusion, :rectified_flow)

# Floor of α = κ in Eq. 12–14. It cancels in the cost (ε̃ᵢ·e^{λᵢ} = σᵢ^{p−1}), but a coarse floor (1e-6) would make
# λ constant where κ < 1e-6 (early times of the Bernoulli path, ᾱ ≈ 0), so the inversion of line 12 would map
# distinct grid points to the same time. For κ(t) = t² on the fine grid (κ ≥ 1e-6), results are unchanged.
const _ATSS_ALPHA_FLOOR = 1f-30

"""
    _bernoulli_kappa(T::Int)

Signal share of the Bernoulli forward process of DDPM as a function of the generation progress τ ∈ [0, 1]:
κ(τ) = ᾱ at diffusion time T(1 − τ), linearly interpolated between integer times, with ᾱ = 1 at time 0.

Parameters:
    - 'T': The total number of diffusion steps.
"""
function _bernoulli_kappa(T::Int)
    ᾱ = vcat(1f0, get_alphas(T))              # ᾱ[k + 1] = ᾱ at diffusion time k
    return function (τ)
        t = clamp(Float32(T) * (1f0 - Float32(τ)), 0f0, Float32(T))
        lo = floor(Int, t)
        hi = min(lo + 1, T)
        w = t - lo
        return (1f0 - w) * ᾱ[lo + 1] + w * ᾱ[hi + 1]
    end
end

"""
    _numeric_derivative(f::Function; h::Float32=1f-3)

Central finite difference of `f` (only stored with the schedule; not used by Algorithm 1).
"""
_numeric_derivative(f::Function; h::Float32=1f-3) =
    τ -> (f(min(Float32(τ) + h, 1f0)) - f(max(Float32(τ) - h, 0f0))) / (min(Float32(τ) + h, 1f0) - max(Float32(τ) - h, 0f0))

"""
    _path_kappa(path::Symbol, κ::Function, κ̇::Function, total_steps::Int)

Signal share κ and its derivative of the method path: the passed κ for :flow_matching, ᾱ of the Bernoulli forward
process for :diffusion, and κ(τ) = τ for :rectified_flow.
"""
function _path_kappa(path::Symbol, κ::Function, κ̇::Function, total_steps::Int)
    path === :flow_matching && return κ, κ̇
    if path === :diffusion
        κ_b = _bernoulli_kappa(total_steps)
        return κ_b, _numeric_derivative(κ_b)
    end
    path === :rectified_flow && return (τ -> Float32(τ)), (τ -> 1f0)
    throw(ArgumentError("ATSS path must be one of $(ATSS_PATHS), got :$path"))
end

Functors.@leaf ATSSScheduler

"""
    _minimize_atss_schedule(T::Int, κ::Function, atss_p::Float32, fine_grid::Vector{Float32}, x_traj::Vector{Matrix{Float32}})

Solves the discrete surrogate integration error objective by coordinate descent over the fine grid to determine the optimal adaptive timesteps.

Parameters:
    - 'T': The number of adaptive inference steps.
    - 'κ': The interpolation function.
    - 'atss_p': The exponent p of the path weight.
    - 'fine_grid': The fine resolution time grid.
    - 'x_traj': The trajectory of states evaluated on the fine grid.
"""
function _minimize_atss_schedule(T::Int, κ::Function, atss_p::Float32,
                                 fine_grid::Vector{Float32}, x_traj::Vector{Matrix{Float32}})

    S = length(fine_grid) - 1
    # R34: every evaluated t is a grid point (candidates and the endpoints 0, 1), so index directly
    # instead of an allocating argmin scan. Results are identical to the previous nearest-point lookup.
    lookup_x(t::Float32) = x_traj[round(Int, t * S) + 1]

    function cost_and_lambda(t_interior::Vector{Float32})
        t_full = vcat(0f0, t_interior, 1f0)
        total = 0f0
        λ_interior = Vector{Float32}(undef, T - 1)
        for i in 1:T
            t_i, t_im1 = t_full[i + 1], t_full[i]
            Δ_i = norm(vec(lookup_x(t_i)) .- vec(lookup_x(t_im1)))   # Eq. 13
            α_ti = κ(t_i)
            σ_ti = max(1f0 - κ(t_i), 1f-6)
            ε̃_i = σ_ti^atss_p / max(α_ti, _ATSS_ALPHA_FLOOR)
            λ_i = log(max(α_ti, _ATSS_ALPHA_FLOOR) / σ_ti)           # Eq. 12
            total += ε̃_i * exp(λ_i) * Δ_i                            # Eq. 14
            if i <= T - 1
                λ_interior[i] = λ_i
            end
        end
        return total, λ_interior
    end

    candidates = fine_grid[2:end-1]

    best_t_interior = fine_grid[round.(Int, range(1, length(fine_grid), length=T+1))][2:T]
    best_cost, best_λ = cost_and_lambda(best_t_interior)

    for j in 1:(T - 1)
        for c in candidates
            trial = copy(best_t_interior)
            trial[j] = c
            sort!(trial)

            # Enforce strict monotonicity to prevent identical timesteps (not in the paper).
            if length(unique(trial)) < length(trial)
                continue
            end

            c_cost, c_λ = cost_and_lambda(trial)
            if c_cost < best_cost
                best_cost, best_λ, best_t_interior = c_cost, c_λ, trial
            end
        end
    end

    return best_λ
end

"""
    _lambda_inverse(κ::Function, λ::Float32; lo::Float32=0f0, hi::Float32=1f0, tol::Float32=1f-6)

Inverts the log-interpolation ratio using bisection to recover the corresponding time step.

Parameters:
    - 'κ': The interpolation function.
    - 'λ': The log-interpolation ratio.
    - 'lo': Lower bound for the bisection search.
    - 'hi': Upper bound for the bisection search.
    - 'tol': Tolerance for the bisection convergence.
"""
function _lambda_inverse(κ::Function, λ::Float32; lo::Float32=0f0, hi::Float32=1f0, tol::Float32=1f-6)
    f(t) = log(max(κ(t), _ATSS_ALPHA_FLOOR) / max(1f0 - κ(t), 1f-6))
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
    _atss_schedule(generator::GraphGenerator, solver::ProblemSolver, T::Int, κ::Function, atss_p::Float32, total_steps::Int)

Executes the ATSS algorithm by sampling a problem instance, evaluating a coupled trajectory on a fine grid, and minimizing the integration error to find adaptive timesteps.

Parameters:
    - 'generator': The graph generator used to sample an instance.
    - 'solver': The solver used to find the target solution.
    - 'T': The number of adaptive inference steps.
    - 'κ': The interpolation function.
    - 'atss_p': The exponent p of the path weight.
    - 'total_steps': Resolution of the fine grid.
"""
function _atss_schedule(generator::GraphGenerator, solver::ProblemSolver,
                         T::Int, κ::Function, atss_p::Float32,
                         total_steps::Int; path::Symbol=:flow_matching)

    @assert T < total_steps "ATSSScheduler: inference_steps ($T) must be < total_steps ($total_steps) for adaptive schedule construction."
    @assert T >= 1 "ATSSScheduler: inference_steps ($T) must be >= 1."

    t_grid = Vector{Float32}(undef, total_steps + 1)   # line 1 (fine substrate, resolution total_steps)
    t_grid[1] = 0f0                                    # line 1: t_0 = 0
    t_grid[total_steps + 1] = 1f0                      # line 1: t_S = 1

    for i in 1:(total_steps - 1)             # line 2
        t_grid[i + 1] = Float32(i) / Float32(total_steps)   # line 3: t_i ← i/S
    end                                       # line 4

    g   = generator()
    x_1, _ = solver(g)                       # target solution, shape (1, n)
    n   = size(x_1, 2)

    x_traj = Vector{Matrix{Float32}}(undef, total_steps + 1)   # x_traj[i+1] = x_{t_i}
    if path === :rectified_flow
        # straight Gaussian path of Rectified Flow: x_t = t·x₁ + (1 − t)·x₀, x₁ ∈ {−1, +1}, x₀ ~ N(0, I)
        x_1s = 2f0 .* Float32.(x_1) .- 1f0
        x_0 = randn(Float32, 1, n)
        for i in 0:total_steps
            t_i = t_grid[i + 1]
            x_traj[i + 1] = t_i .* x_1s .+ (1f0 - t_i) .* x_0
        end
    else
        x_traj[1] = Float32.(rand(Bool, 1, n))   # x_{t_0}, source ~ Unif{0,1}

        committed = falses(1, n)
        for i in 1:total_steps
            t_i   = t_grid[i + 1]
            t_im1 = t_grid[i]

            κ_i, κ_im1 = κ(t_i), κ(t_im1)
            p_commit = clamp((κ_i - κ_im1) / max(1f0 - κ_im1, 1f-6), 0f0, 1f0)

            newly_commit = (.!committed) .& (rand(Float32, 1, n) .< p_commit)
            committed  .|= newly_commit

            x_traj[i + 1] = ifelse.(committed, x_1, x_traj[1])   # coupled: uncommitted entries keep x_0
        end
    end

    λ_interior = _minimize_atss_schedule(T, κ, atss_p, t_grid, x_traj) # line 10

    t_interior = Vector{Float32}(undef, T - 1)
    for i in 1:(T - 1)                                        # line 11
        t_interior[i] = _lambda_inverse(κ, λ_interior[i])  # line 12
    end                                                        # line 13

    return vcat(0f0, t_interior, 1f0)   # line 14: {t_0, t_1, ..., t_T}
end

"""
    ATSSScheduler(::Type{VT}, generator::GraphGenerator, solver::ProblemSolver, total_steps::Int, inference_steps::Int, κ::Function, κ̇::Function; atss_p::Float32=0.5f0, path::Symbol=:flow_matching) where VT

Constructs an ATSSScheduler by fitting a schedule in Float32 on the path of the given method and converting it to
the requested type (Int or Float32). For `path = :diffusion` and `:rectified_flow`, κ is determined by the method
(ᾱ of the Bernoulli forward process, resp. κ(τ) = τ) and the passed κ, κ̇ are not used.

Parameters:
    - 'VT': The target schedule type (Int or Float32).
    - 'generator': The graph generator used to sample an instance.
    - 'solver': The solver used to find the target solution.
    - 'total_steps': Resolution of the fine grid.
    - 'inference_steps': Number of adaptive inference steps.
    - 'κ': The interpolation function.
    - 'κ̇': The derivative of the interpolation function.
    - 'atss_p': The exponent p of the path weight.
    - 'path': The method path to fit on (:flow_matching, :diffusion or :rectified_flow).
"""
function ATSSScheduler(::Type{VT}, generator::GraphGenerator, solver::ProblemSolver,
                        total_steps::Int, inference_steps::Int,
                        κ::Function, κ̇::Function;
                        atss_p::Float32=0.5f0, path::Symbol=:flow_matching) where VT

    if !(VT === Int || VT === Float32)
        error("ATSSScheduler only supports TimeSchedule{Int} or TimeSchedule{Float32}; got $VT")
    end

    κ_path, κ̇_path = _path_kappa(path, κ, κ̇, total_steps)
    steps_raw = _atss_schedule(generator, solver, inference_steps, κ_path, atss_p, total_steps; path=path)
    s_float = ATSSScheduler{Float32}(total_steps, inference_steps, steps_raw, κ_path, κ̇_path, atss_p, path)
    @debug "Steps from ATSSScheduler: " steps_raw

    return VT === Int ? ATSSScheduler(Int, s_float) : s_float
end

# Same default p as the typed constructor (previously 1.0 here vs 0.5 there).
ATSSScheduler(generator::GraphGenerator, solver::ProblemSolver,
              total_steps::Int, inference_steps::Int,
              κ::Function, κ̇::Function; atss_p::Float32=0.5f0, path::Symbol=:flow_matching) =
    ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; atss_p=atss_p, path=path)

ATSSScheduler(::Type{VT}, s::ATSSScheduler{VT}) where {VT} = s

"""
    ATSSScheduler(::Type{Int}, s::ATSSScheduler{Float32})

Converts a fitted Float32 schedule (flow time = generation progress) into integer diffusion times, ensuring strictly increasing steps and repairing duplicates.

Parameters:
    - 's': The Float32 ATSSScheduler to convert.
"""
function ATSSScheduler(::Type{Int}, s::ATSSScheduler{Float32})
    @assert length(s.steps) <= s.total_steps + 1 "ATSSScheduler: inference_steps+1 ($(length(s.steps))) exceeds available integer resolution (total_steps+1 =$(s.total_steps + 1)); cannot produce distinct integer steps."

    T = s.total_steps
    steps = sort(round.(Int, clamp.((1f0 .- s.steps) .* T, 0, T)))              # progress → diffusion time
    steps = accumulate((prev, cur) -> max(prev + 1, cur), steps)              # forward pass, enforce strictly increasing
    steps[end] = min(steps[end], T)                                          # re-clamp in case the forward pass overshot
    steps = reverse(accumulate((prev, cur) -> min(prev - 1, cur), reverse(steps)))  # backward pass, propagate that clamp

    return ATSSScheduler{Int}(T, s.inference_steps, steps, s.κ, s.κ̇, s.atss_p, s.path)
end

"""
    ATSSScheduler(::Type{Float32}, s::ATSSScheduler{Int})

Converts integer diffusion times back into a Float32 schedule representing the generation progress.

Parameters:
    - 's': The Int ATSSScheduler to convert.
"""
function ATSSScheduler(::Type{Float32}, s::ATSSScheduler{Int})
    steps = sort(1f0 .- Float32.(s.steps) ./ Float32(s.total_steps))
    return ATSSScheduler{Float32}(s.total_steps, s.inference_steps, steps, s.κ, s.κ̇, s.atss_p, s.path)
end

Base.convert(::Type{TimeSchedule{VT}}, s::ATSSScheduler) where {VT} = ATSSScheduler(VT, s)
Base.convert(::Type{ATSSScheduler{VT}}, s::ATSSScheduler) where {VT} = ATSSScheduler(VT, s)