# src/problems/misp.jl
#
# Gurobi-based exact solver for the Maximum Independent Set Problem (MISP) delegating ILP formulation to NodeSelectionProblems.jl, and implements greedy decoding based on the sun-23-difusco.pdf paper[cite: 1].

using Graphs
using JuMP
using Gurobi

export MISPSolver

"""
    MISPSolver <: ProblemSolver

A solver that solves the Maximum Independent Set Problem with the usage of Gurobi.
Delegates the actual ILP formulation to `NodeSelectionProblems.ILP!` (via `MIS`),
avoiding duplication of the MIS constraint/objective logic already defined there.
"""
struct MISPSolver <: ProblemSolver
    time_limit::Float64
end

#GR TODO Warum mehrere Gurobi.Envs? Bitte nur eines verwenden, mehrere machen nur Sinn wenn man Threads verwendet. Abgesehen davon gehört das nicht hierher, da das nicht MISP-spezifisch ist. Es ist auch keine "Naturkonstante", dass es genau 2 Lizenzen gibt (wir haben glaube ich am Cluster keine Einschränkungen).
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

"""
    (solver::MISPSolver)(g::Graphs.SimpleGraph)

Solves MISP for `g` via Gurobi. Delegates the ILP formulation
to `ILP!`/`NSPInstance{MIS}`.

Retries once with a fresh `Gurobi.Env()` if the WLS connection drops mid-solve;
a poisoned environment is discarded and replaced in the background
(`GRB_ENV_POOL`) rather than returned to the pool.
#GR TODO Bitte keine derartigen persönlichen Setup-spezifischen Hacks im Package. Bei einem ordentlichen Setup habe ich es nie erlebt, dass man derartige Kunstgriffe benötigt.

Parameters:
    - 'g': The graph instance to solve MISP for.
"""
function (solver::MISPSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    env = take!(GRB_ENV_POOL)
    x_0 = zeros(Float32, 1, n)
    is_poisoned = false

    function build_and_solve(current_env)
        inst = NSPInstance(
            MIS(0), g, n, ne(g),
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
        catch recovery_e  #GR TODO Wenn Gurobi mit einem anderen Fehler terminiert, z.B. weil in der Zeit gar keine gültige Lösung gefunden wurde, dann laden wir doch auch in diesem Zwei, oder? Bei unerwarteten Exceptions von ILP! jedenfalls sauber terminieren! Hier sehe ich die Gefahr, dass nach dem zweiten Versuch eine unülgige bzw. all-Zero Lösung zurückgegeben und mit dieser weitergearbeitet wird!
            @error "WLS recovery failed. Network connection is completely offline." exception=recovery_e
        end
    finally  #GR TODO all das macht in einem allgemeinen Package keinen Sinn, schon gar nicht wenn es hier nur um das MISP geht.
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
        for neighbor in adj[node]
            if in_set[neighbor]
                conflict = true
                break
            end
        end
        if !conflict
            in_set[node] = true
        end
    end

    return in_set
end

"""
    evaluate(::MISPSolver, g::GNNGraph, mask::AbstractVector)

MISP objective: cardinality of the selected (independent) set, i.e. Σᵢ maskᵢ.
"""
function evaluate(::MISPSolver, g::GNNGraph, mask::AbstractVector)
    return Float64(sum(mask))
end