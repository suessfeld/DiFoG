# src/diffusion/scheduler/atss.jl
#
# Implements the Adaptive Time-Step Scheduling (ATSS) strategy from the EFLOCO paper,
# dynamically allocating fine-grained integration steps based on state variations.

using LinearAlgebra: norm
using Functors

export ATSSScheduler

"""
    ATSSScheduler{T} <: Scheduler{T}

Adaptive Time-Step Scheduling (ATSS) strategy.
Implements a non-uniform time-step scheduler that allocates finer resolution to segments of the flow matching probability path with greater state variation. Corresponds to Algorithm 1 in the EFLOCO paper.

Fields:
    - 'total_steps': The total number of fine-grained steps used to evaluate the discrete trajectory for integration error.
    - 'inference_steps': The target number of adaptive steps T to sample during inference.
    - 'steps': The generated adaptive time-step sequence.
    - 'κ': The interpolation function.
    - 'κ̇': The derivative of the interpolation function.
    - 'atss_p': The exponent p controlling the path weight.
"""
struct ATSSScheduler{T} <: Scheduler{T}
    total_steps::Int
    inference_steps::Int
    steps::Vector{T}
    κ::Function
    κ̇::Function
    atss_p::Float32
end

Functors.@leaf ATSSScheduler

"""
    _minimize_atss_schedule(T::Int, κ::Function, atss_p::Float32, fine_grid::Vector{Float32}, x_traj::Vector{Matrix{Float32}})

Solves the discrete surrogate integration error objective to find the optimal log-interpolation ratios.

Parameters:
    - 'T': The target number of adaptive inference steps.
    - 'κ': The interpolation function.
    - 'atss_p': The hyperparameter p for the path weight.
    - 'fine_grid': The fine-grained continuous time grid used for evaluating the trajectory.
    - 'x_traj': The sequence of state matrices sampled along the fine grid.
"""
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
            α_ti = κ(t_i)
            σ_ti = max(1f0 - κ(t_i), 1f-6)
            ε̃_i = σ_ti^atss_p / max(α_ti, 1f-6)
            λ_i = log(max(α_ti, 1f-6) / σ_ti)
            total += ε̃_i * exp(λ_i) * Δ_i
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
            
            # Constraint: Enforce strict monotonicity to prevent identical timesteps. Does not correspond to the paper, however tests showcased that sometimes duplicate timesteps resulted (especially for the INT version) which should not happen in my opinion.
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

Inverts the log-interpolation ratio to recover the continuous time value via bisection.

Parameters:
    - 'κ': The interpolation function.
    - 'λ': The target log-interpolation ratio.
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
    _atss_schedule(generator::GraphGenerator, solver::ProblemSolver, T::Int, κ::Function, atss_p::Float32, total_steps::Int)

Executes the Adaptive Time-Step Scheduling (ATSS) algorithm to generate the inference time steps.

Parameters:
    - 'generator': The graph generator to sample a source instance.
    - 'solver': The exact or heuristic solver to provide the target ground-truth solution.
    - 'T': The desired number of adaptive inference steps.
    - 'κ': The interpolation function.
    - 'atss_p': The path weight exponent.
    - 'total_steps': The resolution of the fine-grained trajectory used for cost evaluation.
"""
function _atss_schedule(generator::GraphGenerator, solver::ProblemSolver,
                         T::Int, κ::Function, atss_p::Float32,
                         total_steps::Int)

    t_grid = Vector{Float32}(undef, total_steps + 1)   # line 1 (fine substrate, resolution total_steps)
    t_grid[1] = 0f0                          # line 1: t_0 = 0
    t_grid[total_steps + 1] = 1f0            # line 1: t_T = 1  (T here = total_steps, the fine grid's own endpoint)

    for i in 1:(total_steps - 1)             # line 2
        t_grid[i + 1] = Float32(i) / Float32(total_steps)   # line 3: t_i ← i/T
    end                                       # line 4

    g   = generator()
    x_1, _ = solver(g)                       # target solution, shape (1, n)
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

    λ_interior = _minimize_atss_schedule(T, κ, atss_p, t_grid, x_traj) # line 10

    t_interior = Vector{Float32}(undef, T - 1)
    for i in 1:(T - 1)                                    # line 11
        t_interior[i] = _lambda_inverse(κ, λ_interior[i])  # line 12
    end                                                    # line 13

    return vcat(0f0, t_interior, 1f0)   # line 14: {t_0, t_1, ..., t_T}
end

"""
    ATSSScheduler(::Type{VT}, generator::GraphGenerator, solver::ProblemSolver, total_steps::Int, inference_steps::Int, κ::Function, κ̇::Function; atss_p::Float32=1.0f0) where VT

Constructs an ATSSScheduler.
"""
function ATSSScheduler(::Type{VT}, generator::GraphGenerator, solver::ProblemSolver,
                        total_steps::Int, inference_steps::Int,
                        κ::Function, κ̇::Function;
                        atss_p::Float32=1.0f0) where VT

    if !(VT === Int || VT === Float32)
        error("ATSSScheduler only supports Scheduler{Int} or Scheduler{Float32}; got $VT")
    end

    steps_raw = _atss_schedule(generator, solver, inference_steps, κ, atss_p, total_steps)

    steps = VT === Int ? round.(Int, clamp.(steps_raw .* total_steps, 0, total_steps)) : steps_raw
    @debug "Steps from ATSSScheduler: " steps

    return ATSSScheduler{VT}(total_steps, inference_steps, steps, κ, κ̇, atss_p)
end

ATSSScheduler(generator::GraphGenerator, solver::ProblemSolver,
              total_steps::Int, inference_steps::Int,
              κ::Function, κ̇::Function; atss_p::Float32=1.0f0) =
    ATSSScheduler(Float32, generator, solver, total_steps, inference_steps, κ, κ̇; atss_p=atss_p)

ATSSScheduler(::Type{VT}, s::ATSSScheduler{VT}) where {VT} = s

function ATSSScheduler(::Type{Int}, s::ATSSScheduler{Float32})
    steps = round.(Int, clamp.(s.steps .* s.total_steps, 0, s.total_steps))
    return ATSSScheduler{Int}(s.total_steps, s.inference_steps, steps, s.κ, s.κ̇, s.atss_p)
end

function ATSSScheduler(::Type{Float32}, s::ATSSScheduler{Int})
    steps = Float32.(s.steps) ./ Float32(s.total_steps)
    return ATSSScheduler{Float32}(s.total_steps, s.inference_steps, steps, s.κ, s.κ̇, s.atss_p)
end

Base.convert(::Type{Scheduler{VT}}, s::ATSSScheduler) where {VT} = ATSSScheduler(VT, s)
Base.convert(::Type{ATSSScheduler{VT}}, s::ATSSScheduler) where {VT} = ATSSScheduler(VT, s)


#GR TODO Was sagen Sie zu folgendem Punkt, den Claude hier anmerkt?
# ATSS scheduling objective is degenerate and pushes all interior steps toward t = 1.**
# `atss.jl:62-64`: `ε̃_i·exp(λ_i)·Δ_i` simplifies to `σ(t_i)^(p−1)·Δ_i`, identically `Δ_i` for the default
# `atss_p = 1`. Because `_atss_schedule` (`atss.jl:160-164`) redraws fresh noise for every uncommitted
# entry at every fine step, `E‖x_{t_i} − x_{t_{i−1}}‖` depends only on the left endpoint, not on the gap,
# so the cost is monotone decreasing in every interior t_j and coordinate descent piles them up near 1.
# The author's own comment in `discrete_flow_matching.jl:52` ("ATSS pushes steps towards 1, which results
# in extreme values") is the symptom, and the `±100` clamp on the velocity scale is the band-aid. After
# Int conversion, clustered steps round to duplicates; `posterior_prob` with `t == s` is an exact
# identity, so ATSS + Bernoulli runs perform far fewer real denoising steps than `inference_steps`. The
# schedule is also fitted from one graph, one solver call and one trajectory (`atss.jl:145-146`), and
# `_lambda_inverse` merely inverts a λ that was computed from the chosen t (identity plus bisection error).
# The surrogate needs a gap-dependent weight (Δλ or ‖Δx‖/Δt) and a coupled trajectory.