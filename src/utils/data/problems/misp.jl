using Graphs
using JuMP
using HiGHS

export MISPSolver

struct MISPSolver <: ProblemSolver
    time_limit::Float64
end

# Keyword constructor with a default of 0.5 seconds
MISPSolver(; time_limit::Float64=0.5) = MISPSolver(time_limit)

function (solver::MISPSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    
    # Use HiGHS which is thread-safe and much faster!
    model = Model(HiGHS.Optimizer)
    set_silent(model) 
    
    # Apply the time limit
    set_time_limit_sec(model, solver.time_limit)
    
    @variable(model, x[1:n], Bin)
    
    for e in edges(g)
        @constraint(model, x[src(e)] + x[dst(e)] <= 1)
    end
    
    @objective(model, Max, sum(x[i] for i in 1:n))
    
    optimize!(model)
    
    x_0 = zeros(Float32, 1, n)
    status = termination_status(model)
    
    # Extract solution if OPTIMAL, or if it hit the TIME_LIMIT but found a feasible answer
    if status == MOI.OPTIMAL || (status == MOI.TIME_LIMIT && has_values(model))
        for i in 1:n
            if value(x[i]) > 0.5
                x_0[1, i] = 1.0f0
            end
        end
    else
        @warn "HiGHS hit the time limit without finding ANY valid solution. Returning empty mask."
    end
    
    return x_0
end