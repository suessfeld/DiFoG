# src/problems/lns.jl
#
# Large Neighborhood Search (LNS) as a labelling/reference solver, for instances on which an exact solver
# does not reach good solutions within its time limit (e.g. MaxCut on larger graphs).
#
# The LNS is the "LNS-rand" baseline of Bresich et al., "Determining Destroy Sets in Large Neighborhood Search
# by Generative Flow Networks" (PPSN 2026; code: gflownets4lns, `get_baseline_lns`), built from the same
# NodeSelectionProblems.jl operators and MHLib.jl:
#   - initial solution: random repair of the empty solution (`construct!`),
#   - destroy: remove d uniformly random nodes of the solution (`baseline_destroy!`, d = 5),
#   - repair: iterated random repair, r independent random fill-ups of the partial solution, keeping the best
#     (`iterative_random_repair!`; r = 2000 for MIS, 1000 for MaxCut, as in the paper),
#   - acceptance: the new solution replaces the current one if it is not worse,
#   - termination: time limit (3 s in the paper).
# Optionally (`init=:greedy`, a deviation from the paper), the LNS starts from the degree greedy solution of the
# problem (the same greedy as the evaluation baseline) instead of a random one; destroy, repair, acceptance and
# time limit are unchanged. On large graphs, 3 s are too short for LNS-rand to get far beyond its random start.
# An LNS provides no bound, so the gap written to the dataset is NaN. With a time limit, the labels depend on the
# speed of the machine and are not bit-for-bit reproducible; the graphs themselves remain seeded.

using Graphs
using JuMP
using MHLib

export LNSSolver

"""
    LNSSolver{B<:ProblemSolver} <: ProblemSolver

Labels instances with the LNS-rand baseline of Bresich et al. (PPSN 2026) instead of an exact solver. The
problem itself (ILP model type, decoder, objective, degree baseline, label symmetry and canonical form) is
taken from the wrapped exact solver `base`, so an `LNSSolver` can be used wherever the base solver is used.

Fields:
    - 'base': The wrapped problem solver defining the problem (`MISPSolver` or `MaxCutSolver`).
    - 'ttime': Time limit of the LNS in seconds.
    - 'n_destroy': Number of nodes removed by each destroy operation (d).
    - 'n_iter_repair': Number of random fill-ups of the iterated random repair (r).
    - 'init': Initial solution, `:random` (LNS-rand) or `:greedy` (degree greedy of the problem).
"""
struct LNSSolver{B<:ProblemSolver} <: ProblemSolver
    base::B
    ttime::Float64
    n_destroy::Int
    n_iter_repair::Int
    init::Symbol
end

"""
    LNSSolver(base::ProblemSolver; ttime=3.0, n_destroy=5, n_iter_repair=default_n_iter_repair(base), init=:random)

Constructs an LNS labelling solver with the settings of the LNS-rand baseline of Bresich et al. by default.

Parameters:
    - 'base': The wrapped problem solver (`MISPSolver` or `MaxCutSolver`).
    - 'ttime': Time limit in seconds (paper: 3 s).
    - 'n_destroy': Destroy set size d (paper: 5).
    - 'n_iter_repair': Repetitions r of the iterated random repair (paper: 2000 for MIS, 1000 for MaxCut).
    - 'init': Initial solution, `:random` (paper) or `:greedy` (start from the degree greedy solution).
"""
function LNSSolver(base::ProblemSolver; ttime::Real=3.0, n_destroy::Int=5,
                   n_iter_repair::Int=default_n_iter_repair(base), init::Symbol=:random)
    base isa LNSSolver && throw(ArgumentError("LNSSolver: base must be an exact solver, not an LNSSolver"))
    ttime > 0 || throw(ArgumentError("LNSSolver: ttime must be > 0, got $ttime"))
    n_destroy >= 1 || throw(ArgumentError("LNSSolver: n_destroy must be ≥ 1, got $n_destroy"))
    n_iter_repair >= 1 || throw(ArgumentError("LNSSolver: n_iter_repair must be ≥ 1, got $n_iter_repair"))
    init in (:random, :greedy) || throw(ArgumentError("LNSSolver: init must be :random or :greedy, got :$init"))
    return LNSSolver{typeof(base)}(base, Float64(ttime), n_destroy, n_iter_repair, init)
end

"""
    default_n_iter_repair(solver::ProblemSolver)

Repetitions r of the iterated random repair used by Bresich et al. for the problem of `solver`.
"""
default_n_iter_repair(::MISPSolver) = 2000
default_n_iter_repair(::MaxCutSolver) = 1000
# evaluated as keyword default before the constructor body, so a nested LNSSolver must reach the ArgumentError there
default_n_iter_repair(solver::LNSSolver) = default_n_iter_repair(solver.base)

"""
    nsp_problem(solver::ProblemSolver)

The NodeSelectionProblems.jl problem type corresponding to `solver` (plain MIS without blocking budget, MaxCut).
"""
nsp_problem(::MISPSolver) = MIS(0)
nsp_problem(::MaxCutSolver) = MAXCUT()

"""
    greedy_mask(solver::ProblemSolver, g::Graphs.SimpleGraph)

The degree greedy solution of the problem as a Bool vector: the evaluation baseline (degree heatmap decoded by the
problem's decoder), i.e. min-degree greedy for MIS and the degree-ordered greedy cut for MaxCut.
"""
function greedy_mask(solver::ProblemSolver, g::Graphs.SimpleGraph)
    gnn = GNNGraph(g)
    return Bool.(vec(decode_heatmap(solver, gnn, degree_baseline_heatmap(solver, gnn))))
end

"""
    construct_from_mask!(sol::NSPSolution, mask::AbstractVector{Bool}, ::Result)

`MHMethod` that sets the solution to the nodes selected in `mask` (for MIS an independent set, for MaxCut one side
of the cut), maintaining the incremental bookkeeping of NodeSelectionProblems.jl.
"""
function construct_from_mask!(sol::NSPSolution, mask::AbstractVector{Bool}, ::MHLib.Result)
    empty!(sol)
    pos = invperm(sol.x)                                  # position of every node in the permutation x
    for u in findall(mask)
        i, j = pos[u], sol.sel + 1                         # move u to the end of the selected prefix
        v = sol.x[j]
        sol.x[i], sol.x[j] = v, u
        pos[v], pos[u] = i, j
        sol.sel += 1
        # MaxCut accepts any side; an MIS mask is independent, so every node is accepted without infeasibility
        MHLib.element_added_delta_eval!(sol; update_obj_val=false, allow_infeasible=sol isa NSPSolution{MAXCUT})
    end
    invalidate!(sol)
end

"""
    (solver::LNSSolver)(g::Graphs.SimpleGraph)

Runs the LNS on `g` until the time limit and returns the best solution found as a canonical (1, n) label,
together with `NaN` as gap (an LNS provides no bound).

Parameters:
    - 'solver': The LNSSolver instance.
    - 'g': The graph instance.
"""
function (solver::LNSSolver)(g::Graphs.SimpleGraph)
    n = nv(g)
    inst = NSPInstance(nsp_problem(solver.base), g, n, ne(g), Ref{Union{Nothing, JuMP.Model}}(nothing), (;))
    sol = NSPSolution(inst)

    construction = solver.init === :greedy ?
        MHMethod("greedy", construct_from_mask!, greedy_mask(solver.base, g)) : MHMethod("con", construct!)
    lns = LNS(sol, [construction],
              [MHMethod("de", baseline_destroy!, solver.n_destroy)],
              [MHMethod("re", iterative_random_repair!, solver.n_iter_repair)];
              consider_initial_sol=true, log=false, lfreq=0, titer=-1, ttime=solver.ttime, tobj=-1.0)
    run!(lns)

    best = lns.scheduler.incumbent
    check(best)

    x_0 = zeros(Float32, 1, n)
    for i in 1:best.sel
        x_0[1, best.x[i]] = 1.0f0
    end
    canonicalize_label!(solver.base, x_0, g)

    return x_0, NaN
end

# Everything problem-specific is taken from the wrapped solver.
label_symmetric(solver::LNSSolver) = label_symmetric(solver.base)
canonicalize_label!(solver::LNSSolver, x::AbstractMatrix, g) = canonicalize_label!(solver.base, x, g)
decode_heatmap(solver::LNSSolver, g::GNNGraph, heatmap::AbstractVector) = decode_heatmap(solver.base, g, heatmap)
evaluate(solver::LNSSolver, g::GNNGraph, mask::AbstractVector) = evaluate(solver.base, g, mask)
degree_baseline_heatmap(solver::LNSSolver, g::GNNGraph) = degree_baseline_heatmap(solver.base, g)
dataset_tag(solver::LNSSolver) = solver.init === :greedy ? "lnsgreedy" : "lns"
base_solver(solver::LNSSolver) = solver.base
