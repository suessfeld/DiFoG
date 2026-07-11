# src/utils/data/problems/misp.jl
using Graphs
using JuMP
using Gurobi

export MISPSolver

const NUM_GUROBI_LICENSES = 2 
const GRB_ENV_POOL = Channel{Gurobi.Env}(NUM_GUROBI_LICENSES)

# Called exactly once from Thesis.__init__()
function _init_gurobi_env!()
    for _ in 1:NUM_GUROBI_LICENSES
        put!(GRB_ENV_POOL, Gurobi.Env())
    end
    return
end

struct MISPSolver <: ProblemSolver
    time_limit::Float64
end

MISPSolver(; time_limit::Float64=0.5) = MISPSolver(time_limit)

function (solver::MISPSolver)(g::AbstractGraph)
    n = nv(g)
    env = take!(GRB_ENV_POOL)
    x_0 = zeros(Float32, 1, n)
    
    # Helper to safely encapsulate the build-and-solve logic
    function build_and_solve(current_env)
        model = Model(() -> Gurobi.Optimizer(current_env))
        set_silent(model)
        set_optimizer_attribute(model, "TimeLimit", solver.time_limit)
        set_optimizer_attribute(model, "Threads", 1)

        # The macro attaches 'x' to the JuMP model dictionary
        @variable(model, x[1:n], Bin)
        for e in edges(g)
            @constraint(model, x[src(e)] + x[dst(e)] <= 1)
        end
        @objective(model, Max, sum(x[i] for i in 1:n))
        optimize!(model)
        return model
    end

    try
        model = nothing
        try
            model = build_and_solve(env)
        catch opt_e
            @warn "Gurobi Token/Network Dropped. Auto-healing WLS environment..." exception=opt_e
            # 1. Discard the poisoned environment by requesting a fresh one
            env = Gurobi.Env()
            
            # 2. Retry the build and solve exactly once with the fresh token
            model = build_and_solve(env)
        end

        status = termination_status(model)
        if status == MOI.OPTIMAL || (status == MOI.TIME_LIMIT && has_values(model))
            x_var = model[:x] # Retrieve the variable reference from the JuMP dictionary
            for i in 1:n
                value(x_var[i]) > 0.5 && (x_0[1, i] = 1.0f0)
            end
        else
            @warn "Gurobi hit the time limit without finding ANY valid solution."
        end
        
        empty!(model)
    finally
        # Put the environment (either the original healthy one, or the newly healed one) 
        # safely back into the pool.
        put!(GRB_ENV_POOL, env)
    end
    
    return x_0
end