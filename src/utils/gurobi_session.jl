# src/utils/gurobi_session.jl
#
# Meta-level Gurobi session management, decoupled from any specific problem (MISP, MaxCut, ...).
# Owns the single shared Gurobi session (one strictly-sequential session, no concurrent env pool) and
# the retry-on-dropped-connection logic, so problem solvers only need to describe how to build and solve
# their own ILP given an `env`.

using Gurobi

export GRB_ENV, GRB_THREADS, with_gurobi_retry

"""
    GRB_ENV

A reference to the single shared Gurobi session used by all problem solvers, ensuring strictly sequential execution.
"""
const GRB_ENV = Ref{Gurobi.Env}()

"""
    GRB_THREADS

A reference to the default thread count Gurobi uses per solve, typically defaulting to half of the available CPU threads.
"""
const GRB_THREADS = Ref{Int}(max(1, Sys.CPU_THREADS ÷ 2))

"""
    _init_gurobi_env!()

Initializes the shared Gurobi environment reference. Typically called once during module initialization.
"""
function _init_gurobi_env!()
    try
        GRB_ENV[] = Gurobi.Env()
    catch e
        @error "Failed to initialize shared Gurobi Environment (GRB_ENV)" exception=e
    end
    return
end

# Errors raised by `ILP!` (NodeSelectionProblems.jl) for a MODEL outcome, not a broken session.
const _MODEL_OUTCOME_MESSAGES = (
    "infeasible or unbounded",
    "time limit reached, no solution found",
    "terminated abnormally",
)

"""
    _is_model_outcome_error(e)

Determines whether a given error represents a valid model solve outcome (such as infeasibility or timeouts) rather than a broken session.

Parameters:
    - 'e': The exception to evaluate.
"""
_is_model_outcome_error(e) =
    e isa InterruptException ||
    (e isa ErrorException && any(m -> occursin(m, e.msg), _MODEL_OUTCOME_MESSAGES))

"""
    with_gurobi_retry(f::Function)

Executes a function using the shared Gurobi environment, automatically discarding the session and retrying once with a fresh environment if a connection error occurs.

Parameters:
    - 'f': The function to execute, which must accept a Gurobi environment as its argument.
"""
function with_gurobi_retry(f::Function)
    try
        return f(GRB_ENV[])
    catch opt_e
        _is_model_outcome_error(opt_e) && rethrow()
        @warn "Gurobi session error (e.g. dropped WLS connection). Discarding environment and retrying once..." exception=opt_e
        try
            GRB_ENV[] = Gurobi.Env()
            return f(GRB_ENV[])
        catch recovery_e
            _is_model_outcome_error(recovery_e) && rethrow()
            @error "Gurobi recovery failed." exception=recovery_e
            return nothing
        end
    end
end