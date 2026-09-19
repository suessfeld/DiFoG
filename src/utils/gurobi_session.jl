# src/utils/gurobi_session.jl
#
# Meta-level Gurobi session management, decoupled from any specific problem (MISP, MaxCut, ...).
# Owns the single shared Gurobi session (see B10/decision: one strictly-sequential session, no
# concurrent env pool) and the retry-on-dropped-connection logic, so problem solvers only need to
# describe how to build and solve their own ILP given an `env`.

using Gurobi

export GRB_ENV, GRB_THREADS, with_gurobi_retry

"""
    GRB_ENV

The single shared Gurobi session, used by every `ProblemSolver`. Initialized in `_init_gurobi_env!`
(called from `Thesis.__init__`).

Deliberately singular, not a pool: strictly sequential use is a meta-level decision (not specific to
any problem) made to avoid the instance-ordering problems concurrent solving introduces in
`create_dataset` — see `seeding.jl`.
"""
const GRB_ENV = Ref{Gurobi.Env}()

"""
    GRB_THREADS

Default thread count Gurobi uses per solve. A meta-level decision (not specific to MISP, MaxCut, or
any other problem), set once here so problem solvers don't each decide it independently.

Defaults to half the machine's CPU threads, reserving the rest for Julia/GC/other work — since the
session architecture is strictly sequential (one `GRB_ENV`, one solve at a time; see
`with_gurobi_retry`), all available Gurobi parallelism goes into this single value.
"""
const GRB_THREADS = Ref{Int}(max(1, Sys.CPU_THREADS ÷ 2))

"""
    _init_gurobi_env!()

Initializes `GRB_ENV`. Called once from `Thesis.__init__`.
"""
function _init_gurobi_env!()
    try
        GRB_ENV[] = Gurobi.Env()
    catch e
        @error "Failed to initialize shared Gurobi Environment (GRB_ENV)" exception=e
    end
    return
end

"""
    with_gurobi_retry(f::Function)

Calls `f(env)` with the current `GRB_ENV[]`. If it throws (e.g. a dropped WLS connection), replaces
`GRB_ENV[]` with a fresh `Gurobi.Env()` in place and retries once. Returns `f`'s result, or `nothing`
if both the original call and the retry failed.

`f` is any problem-specific "build the model for `env` and solve it" closure — this function knows
nothing about MISP, MaxCut, or any other problem; that separation is the point.

Not thread-safe: assumes callers use `GRB_ENV` strictly sequentially (a single Gurobi session, one
solve at a time), not concurrently from multiple tasks.
"""
function with_gurobi_retry(f::Function)
    try
        return f(GRB_ENV[])
    catch opt_e
        @warn "Gurobi Token/Network Dropped. Discarding environment and attempting recovery..." exception=opt_e
        try
            GRB_ENV[] = Gurobi.Env()
            return f(GRB_ENV[])
        catch recovery_e
            @error "WLS recovery failed. Network connection is completely offline." exception=recovery_e
            return nothing
        end
    end
end