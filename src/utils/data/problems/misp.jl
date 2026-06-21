# src/utils/data/problems/misp.jl
using Graphs
using JuMP
using Gurobi

export MISPSolver

const GRB_ENV_REF = Ref{Gurobi.Env}()

# called exactly once, single-threaded, from Thesis.__init__()
function _init_gurobi_env!()
    GRB_ENV_REF[] = Gurobi.Env()
    return
end

struct MISPSolver <: ProblemSolver
    time_limit::Float64
end

MISPSolver(; time_limit::Float64=0.5) = MISPSolver(time_limit)

function (solver::MISPSolver)(g::AbstractGraph)
    n = nv(g)
    model = Model(() -> Gurobi.Optimizer(GRB_ENV_REF[]))
    set_silent(model)
    set_optimizer_attribute(model, "TimeLimit", solver.time_limit)
    set_optimizer_attribute(model, "Threads", 1)

    @variable(model, x[1:n], Bin)
    for e in edges(g)
        @constraint(model, x[src(e)] + x[dst(e)] <= 1)
    end
    @objective(model, Max, sum(x[i] for i in 1:n))
    optimize!(model)

    x_0 = zeros(Float32, 1, n)
    status = termination_status(model)
    if status == MOI.OPTIMAL || (status == MOI.TIME_LIMIT && has_values(model))
        for i in 1:n
            value(x[i]) > 0.5 && (x_0[1, i] = 1.0f0)
        end
    else
        @warn "Gurobi hit the time limit without finding ANY valid solution. Returning empty mask."
    end
    return x_0
end