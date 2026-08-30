using Graphs
using JuMP
using Gurobi

export MISPSolver

"""
A solver that solves the Maximum Independent Set Problem with the usage of Gurobi.
"""
struct MISPSolver <: ProblemSolver
    time_limit::Float64
end


const NUM_GUROBI_LICENSES = 2 
const GRB_ENV_POOL = Channel{Gurobi.Env}(NUM_GUROBI_LICENSES)
const GRB_ENV = Ref{Gurobi.Env}()

function _init_gurobi_env!()
    for _ in 1:NUM_GUROBI_LICENSES
        try
            put!(GRB_ENV_POOL, Gurobi.Env())
        catch e
            @error "Failed to initialize Gurobi Environment during startup" exception=e
        end
    end

    try
        GRB_ENV[] = Gurobi.Env()
    catch e
        @error "Failed to initialize shared Gurobi Environment (GRB_ENV) for NodeSelectionsProblems.jl" exception=e
    end
    return
end

MISPSolver(; time_limit::Float64=0.5) = MISPSolver(time_limit)

function (solver::MISPSolver)(g::AbstractGraph)
    n = nv(g)
    env = take!(GRB_ENV_POOL)
    x_0 = zeros(Float32, 1, n)
    is_poisoned = false
    
    function build_and_solve(current_env)
        model = Model(() -> Gurobi.Optimizer(current_env))
        set_silent(model)
        set_optimizer_attribute(model, "TimeLimit", solver.time_limit)
        set_optimizer_attribute(model, "Threads", 1)

        @variable(model, x[1:n], Bin)
        for e in edges(g)
            @constraint(model, x[src(e)] + x[dst(e)] <= 1)
        end
        @objective(model, Max, sum(x[i] for i in 1:n))
        optimize!(model)
        return model
    end

    try
        model = build_and_solve(env)
        status = termination_status(model)
        if status == MOI.OPTIMAL || (status == MOI.TIME_LIMIT && has_values(model))
            x_var = model[:x]
            for i in 1:n
                value(x_var[i]) > 0.5 && (x_0[1, i] = 1.0f0)
            end
        end
        empty!(model)
    catch opt_e
        @warn "Gurobi Token/Network Dropped. Discarding environment and attempting recovery..." exception=opt_e
        is_poisoned = true
        
        try
            new_env = Gurobi.Env()
            model = build_and_solve(new_env)
            status = termination_status(model)
            if status == MOI.OPTIMAL || (status == MOI.TIME_LIMIT && has_values(model))
                x_var = model[:x]
                for i in 1:n
                    value(x_var[i]) > 0.5 && (x_0[1, i] = 1.0f0)
                end
            end
            empty!(model)
            env = new_env
            is_poisoned = false
        catch recovery_e
            @error "WLS recovery failed. Network connection is completely offline." exception=recovery_e
        end
    finally
        if !is_poisoned
            put!(GRB_ENV_POOL, env)
        else
            @warn "Discarded poisoned Gurobi environment. Re-allocating placeholder to maintain license pool..."
            Threads.@spawn begin
                sleep(5)
                try
                    recovered_env = Gurobi.Env()
                    put!(GRB_ENV_POOL, recovered_env)
                    @info "Successfully restored license slot to pool."
                catch e
                    @error "Could not restore license slot. Retrying connection later." exception=e
                    sleep(10)
                    try
                        recovered_env = Gurobi.Env()
                        put!(GRB_ENV_POOL, recovered_env)
                    catch second_e
                        @error "Critical: Environment slot permanently lost until network restores." exception=second_e
                    end
                end
            end
        end
    end
    
    return x_0
end

"""
    decode_heatmap(::MISPSolver, g::GNNGraph, heatmap::AbstractVector)

Greedy decoding strategy for MISP, as described in Section 3.5 of the
DIFUSCO paper: nodes are visited in descending order of predicted
confidence `heatmap[i]`, and greedily added to the independent set whenever
none of their neighbors are already selected.

Parameters:
    - 'g': The graph representing the problem instance.
    - 'heatmap': The per-node model confidence scores a_i ∈ [0,1].
"""
function decode_heatmap(::MISPSolver, g::GNNGraph, heatmap::AbstractVector)
    num_nodes = g.num_nodes

    sorted_nodes = sortperm(vec(heatmap), rev=true)

    in_set = falses(num_nodes)

    s, t = edge_index(g)
    adj = [Int[] for _ in 1:num_nodes]

    for i in 1:length(s)
        push!(adj[s[i]], t[i])
    end

    for node in sorted_nodes
        conflict = false

        # check if any neighbor is already in the independent set
        for neighbor in adj[node]
            if in_set[neighbor]
                conflict = true
                break
            end
        end

        # if no neighbors are in the set, it's safe to add this node
        if !conflict
            in_set[node] = true
        end
    end

    return in_set
end