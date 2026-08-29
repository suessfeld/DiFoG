using Graphs
using JuMP

export MaxCutSolver

struct MaxCutSolver <: ProblemSolver
    time_limit::Float64
end

MaxCutSolver(; time_limit::Float64=0.5) = MaxCutSolver(time_limit)

function (solver::MaxCutSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    env = take!(GRB_ENV_POOL)
    x_0 = zeros(Float32, 1, n)
    is_poisoned = false

    function build_and_solve(current_env)
        inst = NSPInstance(
            MAXCUT(), g, n, ne(g),
            Ref{Union{Nothing, JuMP.Model}}(nothing),
            (;),
        )
        return ILP!(inst; silent=true, time_limit=solver.time_limit,
                     warn_time_limit=false, env=current_env)
    end

    function apply_solution!(sol)
        for i in 1:sol.sel
            x_0[1, sol.x[i]] = 1.0f0
        end
        
        degs = degree(g)
        max_deg_node = argmax(degs)
        
        if x_0[1, max_deg_node] == 0.0f0
            x_0 .= 1.0f0 .- x_0
        end
    end

    try
        sol, status, rel_gap, solve_time = build_and_solve(env)
        apply_solution!(sol)
    catch opt_e
        @warn "Gurobi Token/Network Dropped. Discarding environment and attempting recovery..." exception=opt_e
        is_poisoned = true

        try
            new_env = Gurobi.Env()
            sol, status, rel_gap, solve_time = build_and_solve(new_env)
            apply_solution!(sol)
            env = new_env
            is_poisoned = false
        catch recovery_e
            @error "WLS recovery failed. Network connection is completely offline." exception=recovery_e
        end
    finally
        if !is_poisoned
            put!(GRB_ENV_POOL, env)
        else
            Threads.@spawn begin
                sleep(5)
                try
                    recovered_env = Gurobi.Env()
                    put!(GRB_ENV_POOL, recovered_env)
                catch e
                    sleep(10)
                    try
                        recovered_env = Gurobi.Env()
                        put!(GRB_ENV_POOL, recovered_env)
                    catch second_e
                    end
                end
            end
        end
    end

    return x_0
end

"""
    decode_heatmap(::MaxCutSolver, g::GNNGraph, heatmap::AbstractVector)

Naive threshold decoding for MAXCUT: each node is assigned to side 1 if its
predicted probability exceeds 0.5, and side 0 otherwise. No neighbor-aware
correction is applied, so the resulting partition is a direct rounding of
the model's heatmap rather than a locally cut-maximizing greedy assignment.

Parameters:
    -'g'       : the graph representing the problem instance (unused, kept
                 for signature consistency with other `decode_heatmap` methods)
    -'heatmap' : per-node model probability scores p_i ∈ [0,1]
"""
function decode_heatmap(::MaxCutSolver, g::GNNGraph, heatmap::AbstractVector)
    return vec(heatmap) .> 0.5f0
end