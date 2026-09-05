# NodeSelectionProblems.jl
#
# Combinatorial optimization problems where a subset of the nodes in a graph is selected.
#
# This includes:
# - Maximum (weighted) independent set problem (MIS)
#   Given an undirected (weighted) graph, find a maximum cardinality subset
#   of nodes where no pair of nodes is adjacent in the graph.
# - Maximum (weighted) clique problem (MCP)
#   Given an undirected (weighted) graph, find a maximum cardinality subset 
#   of nodes where all pairs of nodes are adjacent. 
#   This is the complementary problem of the MIS, i.e., the MC of a graph 
#   is the MIS of the complementary graph. 
# - Maximum cut problem (MAXCUT)
#   Given an undirected (weighted) graph and a partition of its vertices into two 
#   subsets S and T with T = V\S, a cut is defined as the number of edges 
#   between S and T. The MAXCUT problem is to find a set S that maximizes 
#   the cut. 
# - Densest k-subgraph problem (DkS)
#   Given an undirected (weighted) graph and an integer k, find a subset of k nodes
#   that maximizes the number of edges in the induced subgraph.


export NodeSelectionProblem, NSPInstance, NSPSolution,
    MIS, MaxDeg, MCP, MAXCUT, DkS,
    check, construct!, local_improve!, shaking!, baseline_destroy!, random_repair!, 
    ILP!, get_problem, generate_er_graph


"""
    NodeSelectionProblem

Abstract type for optimization problems where a subset of the nodes in a graph is selected.

Concrete subtypes are used within `NSPInstance` to distinguish between different
concrete problems.
"""
abstract type NodeSelectionProblem end

"""
    get_problem(config::Union{Config, NamedTuple})

Return a specific `NodeSelectionProblem` instance based on the `config.problem` symbol.
"""
function get_problem(config::Union{Config, NamedTuple})
    if config.problem == :MIS
        MIS(config.mis_B)
    elseif config.problem == :MaxDeg
        MaxDeg()
    elseif config.problem == :MCP
        MCP()
    elseif config.problem == :MAXCUT
        MAXCUT()
    elseif config.problem == :DkS
        DkS(config.dks_k)
    else
        error("Unknown problem type: $(config.problem)")
    end
end

"""
    NSPInstance{P<:NodeSelectionProblem}

A NodeSelectionProblem instance.

This represents a general node selection problem instance on a graph.
The specific problem is given by the `problem` attribute and its type.

Attributes
- `problem`: concrete problem type, e.g., `MIS`
- `graph`: graph to consider
- `n`: number of nodes
- `m`: number of edges
- `ilp_model`: reference to optional ILP model to solve the instance
- `params`: NamedTuple of additional problem-type parameters for graph-reconstruction
"""
struct NSPInstance{P<:NodeSelectionProblem}
    problem::P
    graph::SimpleGraph{Int}
    n::Int
    m::Int
    ilp_model::Ref{Union{Nothing, Model}}
    params::NamedTuple
end

""" 
    NSPInstance(config::Union{Config, NamedTuple}, graph::SimpleGraph{Int})

Create an NSP instance from the given configuration and graph.
"""
NSPInstance(config::Union{Config, NamedTuple}, graph::SimpleGraph{Int}) =
    NSPInstance(get_problem(config), graph, nv(graph), ne(graph), 
        Ref{Union{Nothing, Model}}(nothing), (;))

"""
    NSPInstance(config:Union{Config, NamedTuple}, rng=nothing)

Generate a random NSP instance based on the provided configuration.
"""
function NSPInstance(config::Union{Config, NamedTuple}, rng::Union{AbstractRNG, Nothing}=nothing)
    # In params, the actually relevant parameters are stored for re-construction of the graph
    problem = get_problem(config)
    if config.graph_type == :er
        params = (; config.problem, config.graph_type, config.n, config.er_p)
        n = params.n isa Int ? params.n : rand(rng, params.n)
        return ERGraphInstance(problem, n, params, rng)
    elseif config.graph_type == :rr
        params = (; config.problem, config.graph_type, config.n, config.rr_k)
        n = params.n isa Int ? params.n : rand(rng, params.n)
        return RRGraphInstance(problem, n, params, rng)
    elseif config.graph_type == :ba
        params = (; config.problem, config.graph_type, config.n, config.ba_m)
        n = params.n isa Int ? params.n : rand(rng, params.n)
        return BAGraphInstance(problem, n, params, rng)
    elseif config.graph_type == :rb_small
        params = (; config.problem, config.graph_type, config.rb_force_mis_size)
        rb_n_groups = rand(rng, 20:25)
        n = rand(rng, 5:12) * rb_n_groups
        p = rand(rng, 0.3:1.0)
    elseif config.graph_type == :rb_large
        params = (; config.graph_type, config.rb_force_mis_size)
        rb_n_groups = rand(rng, 40:55)
        n = rand(rng, 20:25) * rb_n_groups
        p = rand(rng, 0.3:1.0)
    elseif config.graph_type == :rb
        params = (; config.problem, config.graph_type, config.n, config.rb_n_groups, 
            config.er_p, config.rb_force_mis_size)
        rb_n_groups = config.rb_n_groups
        n = params.n
        p = config.er_p
    else
        throw(ArgumentError("Invalid graph type $(config.graph_type)"))
    end
    return RBGraphInstance(problem, n, params, rng)
end

"""
    generate_er_graph(n::Int, p::Float64; rng::Union{Nothing,Random.AbstractRNG}=nothing)

Generates a bare Erdős–Rényi graph G(n, p). This is the single source of truth
for ER generation; both `ERGraphInstance` (NSP path) and `ERGraphGenerator`
(`GraphGenerator` path) delegate here to avoid duplicated `erdos_renyi` calls.

Parameters:
    - 'n': Number of nodes.
    - 'p': Edge probability.
    - 'rng': Optional RNG, forwarded to `Graphs.erdos_renyi` as-is (it natively
      accepts `Union{Nothing,AbstractRNG}`, so no `nothing`-handling shim is needed).
"""
function generate_er_graph(n::Int, p::Float64; rng::Union{Nothing,Random.AbstractRNG}=nothing)
    g = erdos_renyi(n, p; rng)
    return nv(g) == 0 ? SimpleGraph(n) : g   # safety net for degenerate n
end

"""
    ERGraphInstance(problem::NodeSelectionProblem, n, params, rng)

Create a random Erdos-Renyi graph with `n` nodes and edge probability `params.er_p`.

Parameter `params` is a NamedTuple of additional problem-type parameters for 
graph-reconstruction stored in the created instance.
"""
function ERGraphInstance(problem::NodeSelectionProblem, n::Int, params::NamedTuple, rng)
    graph = erdos_renyi(n, params.er_p; rng)
    NSPInstance(problem, graph, n, ne(graph), Ref{Union{Nothing, Model}}(nothing), params)
end


"""
    RRGraphInstance(problem::NodeSelectionProblem, n, params, rng)

Create a random regular graph with `n` nodes having degree `params.rr_k`.

Parameter `params` is a NamedTuple of additional problem-type parameters for 
graph-reconstruction stored in the created instance.
"""
function RRGraphInstance(problem::NodeSelectionProblem, n::Int, params::NamedTuple, rng)
    graph = random_regular_graph(n, params.rr_k; rng)
    NSPInstance(problem, graph, n, ne(graph), Ref{Union{Nothing, Model}}(nothing), params)
end

"""
    BAGraphInstance(problem::NodeSelectionProblem, n, params, rng)

Create a scale-free Barabási-Albert graph with `n` nodes having asymptotic average degree `params.ba_m`.

Parameter `params` is a NamedTuple of additional problem-type parameters for 
graph-reconstruction stored in the created instance.
"""
function BAGraphInstance(problem::NodeSelectionProblem, n::Int, params::NamedTuple, rng)
    graph = barabasi_albert(n, params.ba_m; rng)
    NSPInstance(problem, graph, n, ne(graph), Ref{Union{Nothing, Model}}(nothing), params)
end

"""
    RBGraphInstance(problem::NodeSelectionProblem, n::Int, params::NamedTuple, rng)

Generate a random problem instance according to the RB model as used 
in Zhang et al. (2023) for the MIS.

There are `n` nodes partitioned into `params.rb_n_groups` equally sized groups.
Within each group, all nodes are fully connected.
Then, over a number of iterations, two different groups are selected and their
nodes linked to each other.
Vertices are divided into groups with full intra-group connections, and inter-group edges 
are added randomly.
If `params.rb_force_mis_size` is set to `true`, one node of each group is excluded from 
the process of adding random edges, i.e., the size of an optimal MIS is forced to equal 
the number of groups `params.rb_n_groups`. 
Parameter `params` is a NamedTuple of additional problem-type parameters for 
graph-reconstruction stored in the created instance.
"""
function RBGraphInstance(problem::NodeSelectionProblem, n::Int, 
    params::NamedTuple, rng)
    @assert iszero(n % params.rb_n_groups)
    n_per_group = div(n, params.rb_n_groups)
    a = log(n_per_group) / log(params.rb_n_groups)
    r = -a / log(1 - params.er_p)
    # Create groups (parts) as a 2D array: n groups, each with k nodes
    parts = reshape(1:n, params.rb_n_groups, n_per_group)
    nand_clauses = []

    # Step 1: Fully connect each group (intra-group edges)
    for group in eachrow(parts), i ∈ 1:n_per_group-1, j ∈ i+1:n_per_group
        push!(nand_clauses, (group[i], group[j]))
                push!(nand_clauses, (i, j))
    end

    # Step 2: Add inter-group edges (edges between different groups)
    edge_vec = Vector()
    iterations = floor(Int, r * params.rb_n_groups * log(params.rb_n_groups) - 1)
    idx = params.rb_force_mis_size ? 2 : 1
    for _ in 1:iterations
        # Randomly choose two different groups
        (i, j) = sample(rng, 1:params.rb_n_groups, 2, replace=false)
         
        # Generate all possible edges between these two groups
        all_edges = [(parts[i, a], parts[j, b]) for a ∈ idx:n_per_group, b ∈ idx:n_per_group]
        num_edges = floor(Int, min(params.er_p * length(all_edges), length(all_edges)))
        new_edges = sample(rng, all_edges, num_edges, replace=false)
        # allow duplicates for now, handle them below 
        append!(edge_vec, new_edges)
    end
    
    # Add inter-group edges
    nand_clauses = vcat(nand_clauses, edge_vec)
    
    # Step 3: Create graph from the constructed edges
    graph = SimpleGraph(n)
    for (u, v) ∈ nand_clauses
        @assert u != v
        # `add_edge!()` handles duplicates 
        # edge is only inserted if it does not exist yet 
        add_edge!(graph, u, v)
    end
    NSPInstance(problem, graph, n, ne(graph), Ref{Union{Nothing, Model}}(nothing), params)
end

Base.show(io::IO, inst::NSPInstance) =
    print(io, "$(typeof(inst.problem)): n=$(inst.n), m=$(inst.m)")

"""
    NSPSolution{P<:NodeSelectionProblem}

Solution to an NSP instance.

It is a `SubsetVectorSolution` in which the unselected elements are stored in the
solution vector behind the selected ones. 

Attributes in addition to those needed by `SubsetVectorSolution`:
- `covered`: problem specific; 
    - for MIS: for each node the number of selected neighbor nodes plus one if the node 
        itself is selected; if a budget B is set, then `covered` contains one more element
        that is the total number of covered nodes
    - for MCP: for each node the number of selected neighbor nodes 
    - for MAXCUT: for each node the difference between the number of selected 
        and unselected neighbors 
    - for DkS: for each node the number of selected neighbor nodes
- `n_covered`: number of covered nodes (only for MIS with budget)
"""
mutable struct NSPSolution{P} <: SubsetVectorSolution{Int}
    inst::NSPInstance{P}
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Int}
    sel::Int
    covered::Vector{Int}
    n_covered::Int
end

NSPSolution(inst::NSPInstance) =
    NSPSolution(inst, -1, false, collect(1:inst.n), 0, zeros(Int, inst.n), 0)

function Base.copy!(s1::NSPSolution, s2::NSPSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    copy!(s1.x, s2.x)
    s1.sel = s2.sel
    copy!(s1.covered, s2.covered)
    s1.n_covered = s2.n_covered
end

Base.copy(s::NSPSolution) =
    NSPSolution(s.inst, s.obj_val, s.obj_val_valid, copy(s.x), s.sel, copy(s.covered), 
        s.n_covered)

Base.show(io::IO, s::NSPSolution) =
    println(io, s.x[1:s.sel])

MHLib.calc_objective(s::NSPSolution) = s.sel

function Base.empty!(s::NSPSolution)
    fill!(s.covered, 0)
    s.n_covered = 0
    invoke(Base.empty!, Tuple{SubsetVectorSolution}, s)
end

function MHLib.sort_sel!(s::NSPSolution)
    if s.sel > 1
        sort!(view(s.x, 1:s.sel), alg=Base.Sort.QuickSort)
    end
end

MHLib.check(s::NSPSolution; kwargs...) =
    invoke(check, Tuple{SubsetVectorSolution}, s; kwargs...)

"""
    construct!(::NSPSolution, ::Nothing, result)

`MHMethod that constructs new solution by random initialization.
"""
MHLib.construct!(s::NSPSolution, ::Nothing, result::Result) = initialize!(s)

"""
    local_improve!(::NSPSolution, ::Nothing, result)

`MHMethod` that performs two-exchange local search followed by random fill.
"""
function MHLib.local_improve!(s::NSPSolution, ::Nothing, result::Result)
    if !two_exchange_random_fill_neighborhood_search!(s, false)
        result.changed = false
    end
end

"""
    shaking!(::NSPSolution, par, result)

`MHMethod` that performs shaking by removing `par` random elements followed by random fill.
"""
function MHLib.shaking!(s::NSPSolution, par::Int, ::Result)
    remove_randomly_selected!(s, par)
    fillup!(s)
end

"""
    remove_nodes!(s::NSPSolution, destroy_set)

Remove nodes in `destroy_set` from the solution `s`.
"""
function remove_nodes!(s::SubsetVectorSolution, destroy_set)
    i = 1
    while i ≤ s.sel
        if s.x[i] ∈ destroy_set
            s.x[i], s.x[s.sel] = s.x[s.sel], s.x[i]
            s.sel -= 1
            element_removed_delta_eval!(s, allow_infeasible=true)
        else
            i += 1
        end
    end
end


"""
    baseline_destroy!(sol::NSPSolution, par, ::Result)

Destroy `par` nodes from the current solution randomly.
"""
baseline_destroy!(sol::NSPSolution, par, ::Result) =
    remove_randomly_selected!(sol, par)

"""
    random_repair!(sol::NSPSolution, par, ::Result)

Repair method that fills up the solution with elements in random order.
"""
random_repair!(sol::NSPSolution, par, ::Result) = fillup!(sol)

"""
    iterative_random_repair!(sol::NSPSolution, par, ::Result)

Repair method that applies random repair `par` times.
"""
function iterative_random_repair!(sol::NSPSolution, par, ::Result)
    obj(sol)
    sol_orig = copy(sol)
    sol_best = copy(sol)
    for _ in 1:par
        fillup!(sol)
        is_better(sol, sol_best) && copy!(sol_best, sol)
        copy!(sol, sol_orig)
    end
    copy!(sol, sol_best)
end

"""
    ilp_repair!(sol::NSPSolution, time_limit, ::Result)

Repair method that augments the solution optimally using an ILP solver (Gurobi).

Parameter `time_limit` is the time limit for the ILP solver in seconds.
"""
function ilp_repair!(sol::NSPSolution, time_limit, ::Result)
    # @info "ILP repair called"
    ILP!(sol.inst, sol; silent=true, time_limit, warn_time_limit=false)
    invalidate!(sol)
end

"""
    ILP!(inst::NSPInstance, s::Union{NSPSolution, Nothing}=nothing; silent=false,
        time_limit=inst.config.ilp_time_limit, obj_limit::Union{Nothing,Real}=nothing, 
        warn_time_limit=true)

Solve the NSP instance `inst` using an ILP solver (Gurobi).

If a solution `s` is provided, it is assumed to be a valid partial solution,
e.g., obtained from a destroy operation, and it is augmented in an optimal way.
Thus, the already selected nodes in `s` are fixed to be selected in the ILP.
The returned solution then corresponds to the augmented `s`.

# Further parameters
- `silent`: If set, the solver output is suppressed.
- `time_limit`: The time limit for the solver in seconds.
- `obj_limit`: A limit on the objective value to be reached, 
    i.e., as soon as a feasible solution with at least (most) this objective 
    value is found, the solver terminates.
- `warn_time_limit`: If true, a warning is issued if the time limit is reached
    but a feasible solution has been found.
"""
function ILP!(inst::NSPInstance{P}, s::Union{NSPSolution{P}, Nothing}=nothing; 
        silent=false, time_limit=inst.config.ilp_time_limit, 
        obj_limit::Union{Nothing,Real}=nothing,
        warn_time_limit::Bool=true,
        env::Gurobi.Env=GRB_ENV[]) where P <: NodeSelectionProblem   # NEW — env kwarg
        
    @assert P ∈ (MIS, MCP, MAXCUT, DkS)
    n = inst.n
    t_start = time()
    if P == MAXCUT || P == DkS
        edges_array = collect(edges(inst.graph))
    end

    if isnothing(inst.ilp_model[])
        # construct ILP model   
        inst.ilp_model[] = model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
        silent ? set_silent(model) : unset_silent(model)
        set_time_limit_sec(model, time_limit)
        @variable(model, x[1:n], Bin)

        if P == MCP
            edges_complement = Vector{Graphs.SimpleEdge}()
            for i in 1:n, j in (i+1):n 
                    if !has_edge(inst.graph, i, j)
                        push!(edges_complement, Graphs.SimpleEdge(i, j))
                    end
            end
            @constraint(model, c[e ∈ edges_complement], x[src(e)] + x[dst(e)] ≤ 1)
            @objective(model, Max, sum(x[i] for i ∈ 1:n))

        elseif P == MIS
            if !iszero(inst.problem.B)
                @variable(model, 0.0 ≤ b[i=1:n] ≤ 1.0)
                @constraint(model, calc_b[i ∈ 1:n, j ∈ neighbors(inst.graph, i)], b[i] ≥ x[j])
                @constraint(model, budget, sum(b[i] for i ∈ 1:n) ≤ inst.problem.B)
            end
            @constraint(model, c[e ∈ edges(inst.graph)], x[src(e)] + x[dst(e)] ≤ 1)    
            @objective(model, Max, sum(x[i] for i ∈ 1:n))

        elseif P == MAXCUT
            @variable(model, y[edges_array], Bin)  # edge variables
            @constraint(model, no_endpoint_selected[e = edges_array],
                y[e] <= x[src(e)] + x[dst(e)])
            @constraint(model, both_endpoints_selected[e = edges_array], 
                y[e] <= 2 - x[src(e)] - x[dst(e)])
            @objective(model, Max, sum(y[e] for e ∈ edges_array))

        elseif P == DkS 
            k = inst.problem.k

            @variable(model, y[edges_array], Bin)

            # Exactly k selected nodes
            @constraint(model, cardinality, sum(x[i] for i in 1:n) == k)

            # Edge activation constraints: y[e] can be 1 only if both endpoints are selected
            @constraint(model, edge_act1[e in edges_array], y[e] <= x[src(e)])
            @constraint(model, edge_act2[e in edges_array], y[e] <= x[dst(e)])

            # Objective: maximize the number of induced edges
            @objective(model, Max, sum(y[e] for e in edges_array))

        else 
            error("ILP not implemented for problem type $(typeof(problem))")
        end

        # @info "Finished model construction and starting solving at time: $(time() - t_start)"
    else
        # reuse existing model stored in instance
        model = inst.ilp_model[]
    end
    x = model[:x]  # make model variables accessible
    MOI.set(model, MOI.NumberOfThreads(), 1)
    # add user defined termination criterion on the objective value 
    !isnothing(obj_limit) && set_optimizer_attribute(model, "BestObjStop", obj_limit)

    if !isnothing(s)
        fixed_vars = x[@view s.x[1:s.sel]]
        fix.(fixed_vars, true)
        sol = s
    else
        sol = NSPSolution(inst)
    end

    optimize!(model)
    # @info("Solution summary: $(solution_summary(model))")
    # @info "Finished model solving, time: $(time() - t_start)"
    !silent && println("Finished model solving, time: $(time() - t_start)")

    status = termination_status(model)
    if status == MOI.INFEASIBLE_OR_UNBOUNDED
        error("MIP model is infeasible or unbounded")
    elseif status == MOI.TIME_LIMIT 
        if has_values(model) 
            warn_time_limit && @warn "Gurobi time limit reached, but solution found; gap: \
                $(MOI.get(model, MOI.RelativeGap()))"
        else
            error("Gurobi time limit reached, no solution found.")
        end
    elseif status != MOI.OPTIMAL && status != MOI.OBJECTIVE_LIMIT
        error("MIP solving terminated abnormally: $(termination_status(model))")
    end

    set_from_incidence_vec!(sol, x)

    # has to be done before changing the model with `unfix`
    rel_gap = MOI.get(model, MOI.RelativeGap())
    solve_time = MOI.get(model, MOI.SolveTimeSec())

    if !isnothing(s)
        unfix.(fixed_vars)
    end

    return sol, status, rel_gap, solve_time 
end

function set_from_incidence_vec!(sol::NSPSolution{P}, x::AbstractVector) where 
        P <: NodeSelectionProblem
    inst = sol.inst
    n = inst.n
    sel = 0; not_sel = n+1
    sol.covered .= P == MAXCUT ? -degree(inst.graph) : 0
    for i in 1:n
        if value.(x[i]) ≥ 0.5
            sel += 1
            sol.x[sel] = i
            if P == MIS
                sol.covered[i] += 1
            end
            for v in neighbors(inst.graph, i)
                sol.covered[v] += P == MAXCUT ? 2 : 1
            end
        else
            not_sel -= 1
            sol.x[not_sel] = i
        end
    end
    sol.n_covered = count(>(0), sol.covered)
    sol.sel = sel
    invalidate!(sol)
end

# ------------- MIS ---------------------------------------------------------

"""
    MIS

Maximum independent set problem (MIS).

Given an undirected graph, find a maximum cardinality subset of nodes where
no pair of nodes is adjacent in the graph.

# Elements
- `B::Int`: if non-zero budget on the maximum number of nodes that may be blocked.
"""
struct MIS <: NodeSelectionProblem 
    B::Int 
end

function MHLib.check(s::NSPSolution{MIS}; kwargs...)
    invoke(check, Tuple{SubsetVectorSolution}, s; kwargs...)
    selected = Set(s.x[1:s.sel])
    for e in edges(s.inst.graph)
        if src(e) in selected && dst(e) in selected
            error("Invalid solution - adjacent nodes selected: $(src(e)), $(dst(e))")
        end
    end
    new_covered = zeros(Int, s.inst.n)
    for u in s.x[1:s.sel]
        new_covered[u] += 1
        for v in neighbors(s.inst.graph, u)
            new_covered[v] += 1
        end
    end
    n_covered = count(>(0), new_covered)
    if s.n_covered != n_covered
        error("Invalid number of covered nodes in sol.: $(s.n_covered) versus $(n_covered)")
    end
    if s.covered != new_covered
        error("Invalid covered values in solution: $(s.covered) versus $(new_covered)")
    end
    if !iszero(s.inst.problem.B) && n_covered - s.sel > s.inst.problem.B
        error("Number of blocked nodes exceeds budget: $(n_covered - s.sel) > $(s.inst.problem.B)")
    end
end

MHLib.may_be_extendible(s::NSPSolution{MIS}) = s.sel < length(s.x) && any(iszero, s.covered)

function MHLib.element_removed_delta_eval!(s::NSPSolution{MIS}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel+1]
    s.covered[u] -= 1
    s.n_covered -= 1
    for v ∈ neighbors(s.inst.graph, u)
        s.covered[v] -= 1
        s.covered[v] == 0 && (s.n_covered -= 1)
    end
    if update_obj_val
        s.obj_val -= 1
    end
    return true
end

"""
    n_uncov_neighbors(s::NSPSolution, u::Int)

Count the number of uncovered neighbors of node `u` in solution `s`.
"""
n_uncov_neighbors(s::NSPSolution, u::Int) = 
    count(v -> s.covered[v] == 0, neighbors(s.inst.graph, u))

function MHLib.element_added_delta_eval!(s::NSPSolution{MIS}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)

    @assert !allow_infeasible "Infeasible MIS solutions not supported"
    u = s.x[s.sel]
    B = s.inst.problem.B
    if s.covered[u] == 0 && (B == 0 || n_uncov_neighbors(s, u) < B - (s.n_covered - s.sel))
        # accept
        s.covered[u] += 1
        s.n_covered += 1
        for v ∈ neighbors(s.inst.graph, u)
            s.covered[v] += 1
            s.covered[v] == 1 && (s.n_covered += 1)
        end
        if update_obj_val
            s.obj_val += 1
        end
        return s.covered[u] == 1
    end
    # revert
    s.sel -= 1
    return false
end


# ------------ MaxDeg -------------------------------------------------------

"""
    MaxDeg

Test problem in which a subset of nodes with maximum sum of degrees is to be selected.

The maximum number of nodes that may be selected is given by `num_nodes(::NSPInstance{MaxDeg})`.
"""
struct MaxDeg <: NodeSelectionProblem end

num_nodes(inst::NSPInstance{MaxDeg}) = min(10, inst.n)

MHLib.calc_objective(sol::NSPSolution{MaxDeg}) =
    sum(degree(sol.inst.graph, i) for i ∈ sol.x[1:sol.sel]; init=0)

function MHLib.check(s::NSPSolution{MaxDeg}; kwargs...)
    invoke(check, Tuple{SubsetVectorSolution}, s; kwargs...)
    @assert s.sel ≤ num_nodes(s.inst)
end

MHLib.may_be_extendible(s::NSPSolution{MaxDeg}) = s.sel < num_nodes(s.inst)

function random_repair!(sol::NSPSolution{MaxDeg}, par, ::Result)
    # We fill up to the maximum number of nodes with randomly chosen unselected nodes
    shuffle!(@view sol.x[sol.sel+1:end])
    sol.sel = num_nodes(sol.inst)
    invalidate!(sol)
end

function ILP!(inst::NSPInstance{MaxDeg}, s::Union{NSPSolution{MaxDeg}, Nothing}=nothing; 
        silent=false, time_limit=inst.config.ilp_time_limit, obj_limit=nothing, 
        warn_time_limit::Bool=true)
    # We do not need to solve an ILP, but simply select the nodes with highest degrees
    sol = isnothing(s) ? NSPSolution(inst) : s
    x = sol.x
    unselected = @view x[sol.sel+1:end]
    sort!(unselected, by=i -> degree(inst.graph, i) , rev=true)
    sol.sel = num_nodes(s.inst)
    invalidate!(sol)
    return sol
end

function MHLib.element_removed_delta_eval!(sol::NSPSolution{MaxDeg}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = sol.x[sol.sel+1]
    if update_obj_val
        sol.obj_val -= degree(sol.inst.graph, u)
    end
    return sol.sel ≤ num_nodes(sol.inst)
end

function MHLib.element_added_delta_eval!(s::NSPSolution{MaxDeg}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel]
    if allow_infeasible || s.sel ≤ num_nodes(s.inst)
        # accept
        if update_obj_val
            s.obj_val += degree(s.inst.graph, u)
        end
        return true
    end
    # revert
    s.sel -= 1
    return false
end


# ------------- MCP ---------------------------------------------------------

"""
    MCP

Maximum clique problem (MCP).

Given an undirected graph, find a maximum cardinality subset of nodes where
each pair of nodes is adjacent in the graph.
"""
struct MCP <: NodeSelectionProblem end

function MHLib.check(sol::NSPSolution{MCP}; kwargs...)
    invoke(check, Tuple{SubsetVectorSolution}, sol; kwargs...)
    new_covered = zeros(Int, sol.inst.n)

    MHLib.sort_sel!(sol)
    for i in 1:sol.sel
        for j in (i+1):sol.sel
            if !has_edge(sol.inst.graph, sol.x[i], sol.x[j])
                error("Invalid solution - non-adjacent node(s) selected: $(sol.x[i]), \
                    $(sol.x[j])") 
            end
        end

        for v in neighbors(sol.inst.graph, sol.x[i])
            new_covered[v] += 1
        end
    end

    if sol.covered != new_covered
        error("Invalid covered values in solution: $(s.covered) versus $(new_covered)")
    end
end

MHLib.may_be_extendible(s::NSPSolution{MCP}) =
    s.sel < length(s.x) && any(==(s.sel), s.covered) 

function MHLib.element_removed_delta_eval!(s::NSPSolution{MCP}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel+1]
    for v in neighbors(s.inst.graph, u)
        s.covered[v] -= 1
    end
    if update_obj_val
        s.obj_val -= 1
    end
    return true
end

function MHLib.element_added_delta_eval!(s::NSPSolution{MCP}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel]
    if allow_infeasible || s.covered[u] == (s.sel-1)
        # accept
        for v in neighbors(s.inst.graph, u)
            s.covered[v] += 1
        end
        if update_obj_val
            s.obj_val += 1
        end
        return s.covered[u] == (s.sel-1)
    end
    # revert
    s.sel -= 1
    return false
end


# ------------- MAXCUT ---------------------------------------------------------

"""
    MAXCUT

Maximum cut problem (MAXCUT).

Given an undirected graph, and a partition of its vertices into two 
subsets S and T with T = V-S, a cut is defined as the number of edges between 
S and T. The MAXCUT problem is to find a set S that maximizes the cut. 
"""
struct MAXCUT <: NodeSelectionProblem end

NSPSolution(inst::NSPInstance{MAXCUT}) =
    NSPSolution(inst, -1, false, collect(1:inst.n), 0, -degree(inst.graph), 0)

function Base.empty!(s::NSPSolution{MAXCUT})
    s.sel = 0
    s.covered[:] = -degree(s.inst.graph)
    s.obj_val = 0
    s.obj_val_valid = true
end

function MHLib.calc_objective(s::NSPSolution{MAXCUT})
    selected = falses(nv(s.inst.graph))
    selected[@view s.x[1:s.sel]] .= true
    cut_size = 0
    for u in view(s.x, 1:s.sel), v in neighbors(s.inst.graph, u)
        if !selected[v]
            cut_size += 1
        end 
    end
    # @info selected, cut_size
    return cut_size
end

function MHLib.check(sol::NSPSolution{MAXCUT}; kwargs...)
    invoke(check, Tuple{SubsetVectorSolution}, sol; kwargs...)

    g = sol.inst.graph
    n = nv(g)
    m = ne(g)
    if n > 1 && m > 1 && sol.sel < 1
        error("Empty MAXCUT solution for graph with $n nodes and $m edges.") 
    end

    if n > 1 && m > 1 && sol.sel == n
        error("All nodes selected in MAXCUT solution for graph with $n nodes and $m edges.")
    end

    selected = falses(n)
    selected[@view sol.x[1:sol.sel]] .= true

    new_covered = zeros(Int, sol.inst.n)

    for u in vertices(g)
        nbs = neighbors(g, u)
        nbs_len = length(nbs)
        nbs_sel = count(view(selected, nbs))

        # difference of selected and unselected neighbors 
        new_covered[u] = nbs_sel - nbs_len + nbs_sel        
    end

    if sol.covered != new_covered
        error("Invalid covered values in solution: $(s.covered) versus $(new_covered)")
    end
end

function MHLib.may_be_extendible(s::NSPSolution{MAXCUT})
    s.sel >= length(s.x) && return false
    for u ∈ @view s.x[s.sel+1:end]
        s.covered[u] < 0 && return true
    end
    return false
end

function MHLib.element_removed_delta_eval!(s::NSPSolution{MAXCUT}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel+1]
    selected = view(s.x, 1:s.sel)
    nbs = neighbors(s.inst.graph, u)
    for v in nbs
        s.covered[v] -= 2

        if update_obj_val
            s.obj_val += v in selected ? 1 : -1 
        end
    end

    return true
end

#TODO MB: change to `s.covered[u] <= 0`? 
# a vertex with `covered == 0` can be selected but will not change the objective value 
# keep consistent with `may_be_extendible` 
function MHLib.element_added_delta_eval!(s::NSPSolution{MAXCUT}; 
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel]
    selected = view(s.x, 1:s.sel)
    if allow_infeasible || s.covered[u] < 0
        # accept
        for v in neighbors(s.inst.graph, u)
            s.covered[v] += 2

            if update_obj_val
                s.obj_val += v in selected ? -1 : 1 
            end
        end
        return s.covered[u] < 0
    end
    # revert
    s.sel -= 1
    return false
end

# ------------- DkS ---------------------------------------------------------

"""
    DkS

Densest k-Subgraph problem (DkS).

Given an undirected graph, find a subset of exactly `k` nodes such that the
number of edges in the induced subgraph is maximized.

In our implementation solutions with less or more than `k` nodes are penalized by 
a large negative value proportional to the difference |k - |S||.
"""
struct DkS <: NodeSelectionProblem 
    k::Int
end

const DkS_PENALTY_FACTOR = 10000

"""
    calc_objective(::NSPSolution{DkS})

Calculate the objective value for DkS: the number of edges in the induced subgraph.
"""
function MHLib.calc_objective(s::NSPSolution{DkS})
    G = s.inst.graph
    S = falses(nv(G))
    S[@view s.x[1:s.sel]] .= true
    obj_val = -abs(s.inst.problem.k - s.sel) * DkS_PENALTY_FACTOR  # penalty for infeasibility
    for u in view(s.x, 1:s.sel), v in neighbors(G, u)
        if S[v] && v > u
            obj_val += 1
        end
    end
    return obj_val
end

function MHLib.check(sol::NSPSolution{DkS}; kwargs...)
    # Run generic subset-solution checks (index ranges, sel consistency, etc.)
    invoke(check, Tuple{SubsetVectorSolution}, sol; kwargs...)

    n = sol.inst.n
    G = sol.inst.graph
    k = sol.inst.problem.k

    @assert sol.sel <= k

    is_selected = falses(n)
    is_selected[@view sol.x[1:sol.sel]] .= true

    new_covered = zeros(Int, n)
    for u in sol.x[1:sol.sel]
        new_covered[neighbors(G, u)] .+= 1
    end
    @assert sol.covered == new_covered

    new_obj_val = sum(new_covered[sol.x[1:sol.sel]]; init=0) ÷ 2
    @assert obj(sol) == new_obj_val

    return true
end

MHLib.may_be_extendible(s::NSPSolution{DkS}) = s.sel < s.inst.problem.k

"""
    element_removed_delta_eval!(::NSPSolution{DkS};
                                update_obj_val::Bool=true,
                                allow_infeasible::Bool=false)

Delta evaluation for removing one element from the solution.

Convention (as in MCP): the element that has just been removed from the
selected set is located at position `s.x[s.sel+1]`.

For DkS, every edge between this node and the remaining selected nodes is
removed from the induced subgraph, so the objective value decreases by
`covered[u]`, where `covered[u]` is the number of selected neighbors of `u`.
"""
function MHLib.element_removed_delta_eval!(s::NSPSolution{DkS};
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    u = s.x[s.sel + 1]
    if update_obj_val
        s.obj_val -= s.covered[u]
        s.obj_val += s.inst.problem.k >= s.sel ? -DkS_PENALTY_FACTOR : DkS_PENALTY_FACTOR

    end
    s.covered[neighbors(s.inst.graph, u)] .-= 1
    return true
end


"""
    element_added_delta_eval!(::NSPSolution{DkS};
                              update_obj_val::Bool=true,
                              allow_infeasible::Bool=false)

Delta evaluation for adding one element to the solution.

Convention (as in MCP): the element that has just been added to the
selected set is located at position `s.x[s.sel]`.

For DkS, adding a node u creates new edges to all currently selected neighbors.
The number of such new edges equals `covered[u]` before the update.
We also enforce the cardinality constraint |S| <= k unless infeasibility is allowed.
"""
function MHLib.element_added_delta_eval!(s::NSPSolution{DkS};
        update_obj_val::Bool=true, allow_infeasible::Bool=false)
    k = s.inst.problem.k
    u = s.x[s.sel]
    if !allow_infeasible && s.sel > k
        # Revert the selection
        s.sel -= 1
        return false
    end
    if update_obj_val
        s.obj_val += s.covered[u]
        s.obj_val += k >= s.sel ? DkS_PENALTY_FACTOR : -DkS_PENALTY_FACTOR
    end
    s.covered[neighbors(s.inst.graph, u)] .+= 1
    # Return whether the solution is still considered feasible
    return s.sel <= k || allow_infeasible
end