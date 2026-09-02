# Critical review of `Thesis.jl`

## Context

Review requested of the whole Julia project (no code executed, no tests run). Everything below
comes from reading `src/`, `test/`, `README.md`, `Project.toml`, `.gitignore` and `src/.env`.
Findings are ordered bugs → structure → performance → docs, and within each group by severity.
Line references are clickable.

Existing `#GR TODO` markers are not repeated here except where they interact with a finding.

---

## A. Bugs

### A1. Blocker: the package does not load
[src/Thesis.jl:28](src/Thesis.jl#L28) includes `gaussian_diffusion.jl`, deleted in `c965b68`.
Already marked, listed for completeness — nothing else can be verified until it goes.

### A2. Critical: DFM inference integrates time backwards
`Scheduler` iterates **descending** and skips `steps[end]`
([src/diffusion/scheduler.jl:15-20](src/diffusion/scheduler.jl#L15-L20)): it yields
`steps[L-1], steps[L-2], …, steps[1]`.

That is right for reverse diffusion (Bernoulli seeds `τ_current = total_steps` and ends with an
explicit step to `0`, [bernoulli_diffusion.jl:203-213](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L203-L213)),
but DFM runs on an **ascending** `t: 0 → 1`. `_atss_schedule` returns
`vcat(0f0, t_interior, 1f0)` ([atss.jl:188](src/diffusion/scheduler/atss.jl#L188)), i.e. ascending.
So [discrete_flow_matching.jl:185-191](src/diffusion/methods/flow_matching/discrete_flow_matching.jl#L185-L191)
walks

```
0  →  t_{T-1}  →  t_{T-2}  →  …  →  t_1  →  0
```

One huge forward jump, then every remaining Euler step has `h = τ_next - τ < 0`. Consequences:

- The sampler integrates the velocity field in the wrong direction for `T-1` of its `T` steps.
- The `τ_next >= 1 - 1f-6` terminal branch ([line 141](src/diffusion/methods/flow_matching/discrete_flow_matching.jl#L141))
  is dead: `steps[end] == 1.0` is never yielded.
- The trajectory terminates at `τ = 0` (pure source), not at `τ = 1`.

The final `sigmoid(model(x, t=T))` call at
[lines 193-197](src/diffusion/methods/flow_matching/discrete_flow_matching.jl#L193-L197) masks the
damage — the reported heatmap is one denoiser evaluation, largely independent of the trajectory —
which is likely why this has not shown up as an obvious failure. **All EFLOCO numbers in the thesis
are affected.**

Fix: give `Scheduler` an explicit direction (an ascending iterator for continuous-time methods, or
`Iterators.reverse`), and have DFM consume pairs `(steps[i], steps[i+1])` for `i = 1:L-1`.
`framework.jl` already gets this right — `timesteps` is documented as *"`N+1` ascending timesteps…
first entry is 0 (source), last is 1 (data)"*.

### A3. Average epoch loss is divided by `N+1`
[diffusion_model.jl:73,108,113](src/diffusion_model.jl#L73-L113): `num_batches` starts at `1` and is
incremented once per batch, so after `N` batches it is `N+1`. Every reported/plotted
`avg_loss` is systematically too low, more so for short epochs. The same off-by-one makes the log
line start at "Batch 2" and print `batch_size*num_batches` as the running instance count, which is
wrong whenever the last batch is partial.

### A4. `:rb_small` / `:rb_large` instance generation is broken
[NodeSelectionProblems.jl:122-141](src/utils/data/problems/NodeSelectionProblems.jl#L122-L141): both
branches build `params` **without** `rb_n_groups` and `er_p`, compute `rb_n_groups`, `n`, `p`
into locals that are then discarded, and fall through to `RBGraphInstance`, which immediately reads
`params.rb_n_groups` ([line 204](src/utils/data/problems/NodeSelectionProblems.jl#L204)) and
`params.er_p` ([line 207](src/utils/data/problems/NodeSelectionProblems.jl#L207)). Guaranteed
`FieldError`. `:rb_large` also drops `config.problem` from `params`.

Also in those branches: `rand(rng, 0.3:1.0)` is a range with step 1 whose only element is `0.3` —
it always returns `0.3`. Probably meant `0.3:0.1:1.0` or `0.3 + 0.7*rand(rng)`.

And the `rng::Union{AbstractRNG,Nothing}=nothing` default only survives the `:er` path (Graphs.jl
tolerates `rng=nothing`); `rand(nothing, …)` and `sample(nothing, …)` on the other paths throw.

### A5. RB graph builder pushes bogus edges
[NodeSelectionProblems.jl:213-216](src/utils/data/problems/NodeSelectionProblems.jl#L213-L216):

```julia
for group in eachrow(parts), i ∈ 1:n_per_group-1, j ∈ i+1:n_per_group
    push!(nand_clauses, (group[i], group[j]))
            push!(nand_clauses, (i, j))     # ← stray, and mis-indented
end
```

The second `push!` inserts the raw **within-group indices** as node ids, adding a clique over nodes
`1:n_per_group` on top of the intended structure. This silently changes the instance distribution
(and, for MIS, the forced optimum that `rb_force_mis_size` is supposed to guarantee).

### A6. `ILP!` default argument references a non-existent field
[NodeSelectionProblems.jl:431](src/utils/data/problems/NodeSelectionProblems.jl#L431):
`time_limit = inst.config.ilp_time_limit`. `NSPInstance` has fields
`problem, graph, n, m, ilp_model, params` — no `config`
([lines 84-91](src/utils/data/problems/NodeSelectionProblems.jl#L84-L91)). Any call to `ILP!` that
omits `time_limit` throws. Same stale default on the `MaxDeg` method
([line 697](src/utils/data/problems/NodeSelectionProblems.jl#L697)).

### A7. `ILP!` for `MaxDeg` dereferences a possibly-`nothing` argument
[NodeSelectionProblems.jl:700-704](src/utils/data/problems/NodeSelectionProblems.jl#L700-L704):
`sol = isnothing(s) ? NSPSolution(inst) : s`, then `sol.sel = num_nodes(s.inst)` — should be
`sol.inst`. Throws on the no-solution-supplied path.

### A8. Three error paths that themselves throw `UndefVarError`
- [NodeSelectionProblems.jl:493](src/utils/data/problems/NodeSelectionProblems.jl#L493):
  `typeof(problem)` — no `problem` in scope (`inst.problem`).
- [NodeSelectionProblems.jl:765](src/utils/data/problems/NodeSelectionProblems.jl#L765) (MCP `check`)
  and [line 868](src/utils/data/problems/NodeSelectionProblems.jl#L868) (MAXCUT `check`):
  `$(s.covered)` — the argument is named `sol`.

These fire exactly when a consistency check has failed, i.e. they replace the diagnostic you need
with a spurious one.

### A9. Exported names that do not exist
[decoding_utils.jl:4](src/utils/decoding_utils.jl#L4) exports `greedy_decode_mis` and
`greedy_decode_maxcut`; neither is defined anywhere in the repo. `using Thesis; greedy_decode_mis`
gives `UndefVarError`. Leftovers from the pre-`decode_heatmap` refactor.

### A10. Silently degenerate ground-truth labels
[misp.jl:60-68](src/utils/data/problems/misp.jl#L60-L68): if the status is neither `OPTIMAL` nor
`TIME_LIMIT`-with-values, `x_0` stays **all zeros** and is returned as if it were the label. The
`RuntimeDataset` worker cannot tell the difference and trains on it. `maxcut.jl` is worse: it binds
`status` and `rel_gap` at [line 45](src/utils/data/problems/maxcut.jl#L45) and never looks at them.

With `time_limit = 0.5` (the default, and 0.2 in the README example), a non-trivial fraction of
labels are time-limited and hence *suboptimal but unmarked*. For a thesis this needs at minimum to
be measured: log the gap distribution, or reject instances above a gap threshold.

### A11. Lost Gurobi license slot deadlocks training silently
[misp.jl:95-111](src/utils/data/problems/misp.jl#L95-L111): if both delayed `Gurobi.Env()` retries
fail, the slot is never restored. Every `RuntimeDataset` worker then blocks forever in
`take!(GRB_ENV_POOL)` ([misp.jl:41](src/utils/data/problems/misp.jl#L41)). Training stalls with no
error and no timeout — the exact failure mode the `catch`/resume loop was written to avoid.
The MaxCut copy is worse still: its inner `catch second_e` body is **empty**
([maxcut.jl:74-75](src/utils/data/problems/maxcut.jl#L74-L75)), so the slot vanishes without a log.

### A12. Model is saved twice
[test_utils.jl:262-268](test/test_utils.jl#L262-L268) and
[test_utils.jl:340-346](test/test_utils.jl#L340-L346) are the same block. Doubles the write and
emits "Model successfully saved" twice per run (four times, counting the one inside `save_model`).

### A13. DkS removal penalty has the wrong sign at `sel == k`
[NodeSelectionProblems.jl:1002](src/utils/data/problems/NodeSelectionProblems.jl#L1002):
`k >= s.sel ? -P : +P`. The true penalty delta is `-P` for `sel_new < k` and `+P` for
`sel_new >= k`; the boundary `sel_new == k` takes the wrong branch. (The `added` counterpart at
[line 1035](src/utils/data/problems/NodeSelectionProblems.jl#L1035) is correct.)

### A14. `latex_escape` double-escapes its own output
[test_utils.jl:357-368](test/test_utils.jl#L357-L368): `\` → `\textbackslash{}` runs *before* the
loop that maps `{` → `\{` and `}` → `\}`, so a backslash in a `run_id` or type name emits
`\textbackslash\{\}` — broken LaTeX. Escape braces first, or substitute a placeholder.

### A15. Dead guard in the ER generator
[er.jl:23-25](src/utils/data/graphs/er.jl#L23-L25): `nv(g) == 0` can never hold — `erdos_renyi(v, p)`
always returns `v` vertices. The intended guard is presumably `ne(g) == 0`.

### A16. Cached ILP model ignores a changed `time_limit`
[NodeSelectionProblems.jl:443-500](src/utils/data/problems/NodeSelectionProblems.jl#L443-L500):
`set_time_limit_sec` is inside the construction branch only. On reuse of `inst.ilp_model[]` the
`time_limit` argument is silently dropped.

---

## B. Structural weaknesses

### B1. The `Scheduler` iterator contract is violated
[scheduler.jl:10-20](src/diffusion/scheduler.jl#L10-L20) defines
`length(s) == length(s.steps) - 1`, an iterator that yields `steps[end-1:-1:1]`, **and**
`Base.collect(s) = s.steps` — three mutually inconsistent views of the same object. `collect` is
documented to return the iterator's elements; here it returns a different length, in the opposite
order, including an element iteration never produces. Anything generic (`sum`, `maximum`,
`[x for x in s]`, `Iterators.*`) silently disagrees with `collect`, and the test suite bakes the
inconsistency in ([test_schedulers.jl:19-21](test/cases/test_schedulers.jl#L19-L21) asserts
`length(collect(s)) == inference_steps` while
[line 15](test/cases/test_schedulers.jl#L15) asserts `length(s) == inference_steps - 1`).

Rename `collect` to `get_steps` (which already exists and is unused,
[scheduler.jl:23](src/diffusion/scheduler.jl#L23)), or make the iterator yield the pairs the
samplers actually need — `(τ_from, τ_to)`. The latter would also fix A2 structurally, since both
samplers currently reconstruct the pairing by hand with different conventions.

### B2. Abstract field types throughout the hot path
`BernoulliDiffusion.scheduler::Scheduler{Int}`
([bernoulli_diffusion.jl:16](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L16)),
`DiscreteFlowMatching.scheduler::Scheduler{Float32}` plus `κ::Function`, `κ̇::Function`
([discrete_flow_matching.jl:21-26](src/diffusion/methods/flow_matching/discrete_flow_matching.jl#L21-L26)),
`ATSSScheduler.κ/κ̇::Function` ([atss.jl:20-27](src/diffusion/scheduler/atss.jl#L20-L27)),
`NSPInstance.params::NamedTuple`.

Every field access is a boxed load with an unpredictable return type. `diffusion.κ.(τ_r)` inside a
gradient tape is a dynamic dispatch per call, and it prevents the broadcast from specialising. Make
them parametric:

```julia
struct DiscreteFlowMatching{S<:Scheduler{Float32}, K, Kd} <: Diffusion
    scheduler::S; λ_cons::Float32; κ::K; κ̇::Kd
end
```

### B3. `RuntimeDataset` leaks unstoppable workers and misreports its concurrency
[runtime_dataset.jl:51-69](src/utils/data/runtime_dataset.jl#L51-L69): the constructor spawns
`max_concurrent_solves` tasks in `while true` loops with no shutdown path. The pipeline builds two
datasets with `workers = 12` ([test_utils.jl:61,80](test/test_utils.jl#L61)), so 24 permanent tasks
outlive the run. Only 2 can ever make progress — `NUM_GUROBI_LICENSES = 2`
([misp.jl:16](src/utils/data/problems/misp.jl#L16)) — so 22 sit blocked on `take!` forever, and
`max_concurrent_solves` does not actually control concurrency.

Also: the `catch e` swallows `InterruptException`, so Ctrl-C inside a worker just logs and resumes;
and the per-worker counter `n` ([line 53](src/utils/data/runtime_dataset.jl#L53)) means both workers
emit `runtime_inst_0.txt`, `runtime_inst_1.txt`, … colliding names.

Suggest: bound the workers by the license pool, take a `Channel` you can `close!`, and register a
shutdown (`close(buffer)` + a stop flag) so a finished run releases its threads.

### B4. Two parallel implementations of everything data-related
Already flagged by the `#GR TODO`s, but the cost is concrete: `MISPSolver`/`MaxCutSolver`
duplicate ~55 lines of env-pool + WLS-recovery logic between
[misp.jl:39-116](src/utils/data/problems/misp.jl#L39-L116) and
[maxcut.jl:15-82](src/utils/data/problems/maxcut.jl#L15-L82), and they have already **diverged** —
maxcut's recovery is silent (A11). This should be one
`with_gurobi_env(f)` helper. Likewise `ERGraphGenerator`/`BasicGraphGenerator` vs.
`ERGraphInstance`/`RRGraphInstance`/`BAGraphInstance`/`RBGraphInstance`: the second family (RR, BA,
RB) is unreachable from the `GraphGenerator` interface the pipeline actually uses, so the thesis can
only run on ER and the ad-hoc `BasicGraphGenerator`.

### B5. Time-direction and time-type conventions are implicit and conflicting
`Scheduler{Int}` for Bernoulli vs `Scheduler{Float32}` for DFM, descending vs ascending time,
`steps[end]` meaning `total_steps` in one and `1.0` in the other. None of this is enforced by the
type system or documented outside `CLAUDE.md`; A2 is the direct consequence. `framework.jl`'s single
normalised `t ∈ [0,1]` timeline is the right target.

### B6. No reproducibility
No generator, solver, corruption step or sampler takes an RNG. Combined with `RuntimeDataset`'s
`getindex` ignoring its index ([runtime_dataset.jl:76](src/utils/data/runtime_dataset.jl#L76)),
no run in the thesis can be reproduced — not the instances, not the training order, not the samples.
For a thesis this is worth fixing before the next round of experiments: thread an
`rng::AbstractRNG` through `GraphGenerator`, and use `CUDA.rand!` with a seeded `CURAND` state.

### B7. Evaluation methodology
- **The headline metric is only meaningful for MIS.** `total_model_score` /`total_true_score`
  ([test_utils.jl:143,158](test/test_utils.jl#L143-L158)) count *selected nodes*. For MaxCut the
  objective is the cut value, not `|S|` — two partitions with identical `|S|` can have wildly
  different cuts, so the reported "Approximation Ratio (%)" for MaxCut measures nothing. For MaxDeg
  the decoder picks exactly `k = |argmax degree|` nodes by construction, so the ratio is pinned
  near 100% regardless of model quality. Score via `decode_heatmap` → problem objective
  (`MHLib.calc_objective` on an `NSPSolution` already computes cut size and MIS size).
- **No validation set and no meaningful epochs.** Because `getindex` ignores its index, each "epoch"
  consumes `num_train` *fresh* instances. There is no held-out set, so overfitting is unobservable
  and early stopping impossible.
- **Test set is drawn from the training distribution.** Same generator, same size range. No
  out-of-distribution / larger-graph generalisation experiment, which is the standard claim in the
  DIFUSCO line of work.
- **Single run, no seeds, no error bars.**

### B8. BatchNorm makes inference batch-dependent
All `BatchNorm`s use `track_stats=false`
([AGNN.jl:17,41-42](src/gnn/models/AGNN.jl#L41-L42)), so normalisation always uses the current
batch's statistics — in training *and* at inference. `Flux.testmode!(model.gnn)`
([test_utils.jl:99](test/test_utils.jl#L99)) is therefore a no-op, and a given instance's heatmap
depends on which other instances happen to share its batch. That confounds the
repeated-inference-stability metric ([test_utils.jl:213-253](test/test_utils.jl#L213-L253)), which is
measured on a batch of one while the main evaluation runs at `batch_size = 16`.

### B9. CUDA is hard-wired; there is no CPU path
14 unconditional `CUDA.rand`/`CUDA.zeros`/`CUDA.fill` calls in `src/`, plus
`|> Flux.gpu` inside `get_timestep_embedding`
([diffusion_model.jl:188](src/diffusion_model.jl#L188)) and `q`
([bernoulli_diffusion.jl:58](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L58)). This makes
the package untestable in CI, unusable for anyone without a GPU, and makes `Flux.gpu`/`Flux.cpu`
meaningless as a switch. `similar(x, …)` + `rand!`/`zero` would be device-agnostic at no cost.

### B10. Timestep-embedding assembly is duplicated three times
[diffusion_model.jl:30-34](src/diffusion_model.jl#L30-L34) defines `_embed_timesteps`, then
[diffusion_model.jl:163-167](src/diffusion_model.jl#L163-L167) re-inlines the identical logic
instead of calling it, and
[discrete_flow_matching.jl:235-239](src/diffusion/methods/flow_matching/discrete_flow_matching.jl#L235-L239)
implements it a third time as `_build_model_input`. Meanwhile `AGNN` immediately *splits* the
`vcat` back apart ([AGNN.jl:101-102](src/gnn/models/AGNN.jl#L101-L102)). The `in_dim =
time_dim + feature_dim` contract is spread across three files with no single owner — passing
`(x_t, t_emb)` as two arguments would remove the packing, the unpacking and the assertion.

### B11. Dead code
`adjacency_list_to_matrix` ([math_utils.jl:14](src/utils/math_utils.jl#L14)) — used only by its own
test; `get_steps`; `infer` ([diffusion_model.jl:130](src/diffusion_model.jl#L130), documented in the
README but unused, and the pipeline calls `sample_heatmap` directly); `κ̇` is stored by
`ATSSScheduler` and threaded through `_atss_schedule` but **never called**; the `VT` type parameter
of `_atss_schedule` is unused; `ε̃`, `λ`, `L`, `Δ` are computed over the whole fine grid at
[atss.jl:161-179](src/diffusion/scheduler/atss.jl#L161-L179) and then discarded (see C2);
`abstract type Config` ([Thesis.jl:49](src/Thesis.jl#L49)) has no subtype; `INSTANCES_PATH` and
`MODELS_PATH` in `src/.env` are read by nothing.

### B12. Device movement relies on undocumented Functors behaviour
`DiffusionModel` and the schedulers are not `@layer`-annotated; they are moved by Functors' *default*
struct recursion, which rebuilds via `constructorof(T)(fields...)`. That is why the
"Catch GPU arrays during `Flux.gpu` traversal" constructors exist
([linear.jl:62-64](src/diffusion/scheduler/linear.jl#L62-L64) and siblings). It works, but it is
fragile in one concrete way: `κ` as a **closure** (rather than a top-level function) has fields, so
Functors will recurse into it and try to reconstruct it — `DiffusionModel(...) |> Flux.gpu` would
fail. Worth a `Flux.@leaf`/`@layer` annotation and a comment.

---

## C. Inefficiencies

### C1. ATSS schedule fitting is quadratic in both grid sizes
[atss.jl:69-85](src/diffusion/scheduler/atss.jl#L69-L85) runs `(T-1) × (total_steps-1)` calls to
`cost_and_lambda`, each looping `T` times and calling `lookup_x` twice — and `lookup_x`
([line 44](src/diffusion/scheduler/atss.jl#L44)) does `argmin(abs.(fine_grid .- t))`, i.e. an
allocation and a full O(total_steps) scan **per lookup**, followed by an O(n) `norm`.

Total ≈ `T² · total_steps² · n`. At the paper's settings (`T = 50`, `total_steps = 1000`,
`n = 300`) that is ~10¹² operations for a single scheduler construction. Even the unit test
(`T = 10`, `total_steps = 100`) does ~10⁷.

Two cheap fixes: `lookup_x` is an index computation, `x_traj[round(Int, t*total_steps) + 1]`
(O(1), no allocation); and `Δ` between fine-grid points can be precomputed once into a
`total_steps × total_steps` table, or the sum restructured so only the changed coordinate's two
adjacent terms are recomputed per trial (turning the inner loop from O(T) into O(1)).

### C2. The ATSS λ round-trip is an identity
`cost_and_lambda` returns `λ_i = log(κ(t_i)/(1-κ(t_i)))` for the best interior points
([atss.jl:55](src/diffusion/scheduler/atss.jl#L55)), and `_atss_schedule` then inverts each one by
bisection to recover `t_i` ([atss.jl:183-186](src/diffusion/scheduler/atss.jl#L183-L186)).
Since `_lambda_inverse` is the exact inverse of that map, this reconstructs the values it already
had — at the cost of ~20 bisection iterations each and a precision loss of `tol = 1f-6`, plus a real
failure mode: the `max(·, 1f-6)` clamps saturate λ for `t` near 0 or 1, and the inverse of a
saturated λ is not the original `t`. Return `best_t_interior` directly.

The `ε̃ / λ / L / Δ` loop at [atss.jl:161-179](src/diffusion/scheduler/atss.jl#L161-L179) computes
the full Eq. 14 cost table over the fine grid and then never uses any of it.

### C3. The ATSS schedule is fitted to a single random instance
[atss.jl:139-159](src/diffusion/scheduler/atss.jl#L139-L159) draws **one** graph, solves it once, and
samples **one** stochastic trajectory `x_traj` (the commit process is random). The whole schedule —
used for every subsequent inference — is fitted to that one sample path. Average `Δ` over several
instances and several trajectories; it is far cheaper than the search itself and removes an obvious
source of variance in the results.

### C4. β/ᾱ recomputed and re-uploaded on every call
`get_betas` allocates a `range` and `get_alphas` a `cumprod` on **every** invocation
([bernoulli_diffusion.jl:26-44](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L26-L44)),
and `q` then does `|> Flux.gpu` on the whole `αs` vector for every corruption
([line 58](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L58)), while `posterior_prob`
does two 1-element host→device transfers per reverse step
([lines 131-136](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L131-L136)). With
`total_steps = 1000` and 50 inference steps that is 100 tiny synchronising transfers per sample plus
a 1000-element `cumprod` on the host, per training batch. Precompute `ᾱ` once into the struct as a
device array.

### C5. AGNN broadcasts the timestep embedding to every edge, in every layer
[AGNN.jl:111,117](src/gnn/models/AGNN.jl#L111-L117): `t_emb_edges = t_emb[:, s]` materialises a
`time_dim × num_edges` matrix, and each of the `L` layers pushes it through a 2-layer MLP over all
`E` columns. But `t` is constant per graph — there are only `num_graphs` distinct values. Project
once per graph (a `time_dim × B` matrix), then gather to edges, or better fold it in as a bias:
`L × 2 × E × d` FLOPs collapse to `L × 2 × B × d`. At `E ≈ 20k`, `B = 16`, `L = 12`, `d = 256` this
is a three-orders-of-magnitude reduction on a term that plausibly dominates the forward pass.

Related: `get_timestep_embedding` moves `freqs` to the device on every call
([diffusion_model.jl:188](src/diffusion_model.jl#L188)) — hoist it.

### C6. Smaller items
- `decode_heatmap(::MISPSolver, …)` rebuilds a `Vector{Vector{Int}}` adjacency list per instance
  ([misp.jl:138-142](src/utils/data/problems/misp.jl#L138-L142)); `neighbors(g, node)` on the
  underlying graph avoids the allocation entirely. It also only walks `s → t`, which is correct only
  because `GNNGraph` stores undirected graphs bidirected — an undocumented dependency.
- `BasicGraphGenerator` materialises all O(n²) non-tree pairs and `shuffle`s the whole vector to take
  a prefix ([basic.jl:43-57](src/utils/data/graphs/basic.jl#L43-L57)); `StatsBase.sample(…, k;
  replace=false)` or reservoir sampling is O(k).
- The prefetch `Channel(3, spawn=true)` ([diffusion_model.jl:75](src/diffusion_model.jl#L75)) is
  untyped, so every batch is boxed; and it is rebuilt per epoch.
- `Flux.DataLoader(…, shuffle=true)` ([test_utils.jl:62](test/test_utils.jl#L62)) is a no-op given
  `getindex`'s behaviour — misleading, drop it or document it at the call site.
- `Flux.params` ([test_utils.jl:77](test/test_utils.jl#L77)) still resolves in Flux 0.16 but only via
  `deprecations.jl`. Use `sum(length, Flux.trainables(model.gnn))`.

---

## D. Documentation problems

### D1. README describes a source layout that does not exist
- Module tree says `tests/` with `src/tests/cases/*`
  ([README.md:88-98](README.md#L88-L98)); reality is `test/runtests.jl`, `test/test_utils.jl`,
  `test/cases/`.
- The **example pipeline does not run**: `include(joinpath(pkgdir(Thesis), "src", "tests",
  "test_utils.jl"))` ([README.md:116](README.md#L116)) points at a non-existent path.
- The Testing section ([README.md:225-230](README.md#L225-L230)) tells the reader to run
  `julia --project=. src/tests/testrunner.jl`; the file is `test/runtests.jl`.
- `test_graph_generator.jl` ([README.md:96](README.md#L96)) is really `test_graph_generators.jl`.
- `.gitignore` carries the same stale paths (`src/tests/pipelines`,
  `src/tests/cases/cluster/*`), and `models/` — which the pipeline writes `.jld2` weights into — is
  **not** ignored.

### D2. README contradicts itself on what is maintained
[README.md:36](README.md#L36) lists `GaussianDiffusion` as an implementation of `Diffusion`
(deleted in `c965b68`) and marks it "(unmaintained)", while
[README.md:64](README.md#L64) marks **`bernoulli_diffusion.jl`** "(unmaintained)" — that is the
primary DIFUSCO implementation and the one the README's own example uses.

### D3. "the rest of the script is unchanged" is false
[README.md:169](README.md#L169) says swapping `BernoulliDiffusion(scheduler)` for
`DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇)` leaves the rest unchanged. It does not: the example
constructs `CosineScheduler(1000, 50)`, which defaults to `Scheduler{Int}`
([cosine.jl:19](src/diffusion/scheduler/cosine.jl#L19)), and `DiscreteFlowMatching` requires
`Scheduler{Float32}`. The scheduler-eltype ↔ method coupling — the single sharpest footgun in the
API — appears nowhere in the README, including in the "Extending the framework" section
([README.md:217](README.md#L217)).

The same paragraph claims a common interface of `corrupt_batch`, `p`, `sample_heatmap`; the two `p`
methods actually have incompatible signatures (`::Int, ::Int` vs `::Real, ::Real`).

### D4. The heatmap-quality metrics are described wrongly everywhere they appear
`heatmap_decisiveness` ([decoding_utils.jl:36-41](src/utils/decoding_utils.jl#L36-L41)) returns
`(1 - mean|2p-1|, mean|2p-1|)`. But:
- the report labels the first value **"Avg Deviation from 0.5 $\bar{H}$"**
  ([test_utils.jl:325](test/test_utils.jl#L325)) — it is one *minus* the mean deviation;
- it labels the second **"Avg Decisiveness Score $1 - \bar{H}/\ln 2$"**
  ([test_utils.jl:326](test/test_utils.jl#L326)) — that formula is a normalised binary entropy,
  which is not what is computed;
- the test asserts `mean_entropy <= Float32(log(2))` and comments about "eps clamping"
  ([test_decoding_utils.jl:47-54](test/cases/test_decoding_utils.jl#L47-L54)) — there is no entropy
  and no clamping in the implementation.

Three places describe a binary-entropy quantity the code does not compute; these labels would go
straight into the thesis. Either implement `H(p) = -p log p - (1-p) log(1-p)` or rename the outputs
and fix all three descriptions.

### D5. Docstring / signature mismatches
- `posterior_prob` ([bernoulli_diffusion.jl:117](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L117))
  and `p` ([line 162](src/diffusion/methods/diffusion/bernoulli_diffusion.jl#L162)) both show a
  signature missing the `s` argument that the functions actually take.
- `q`'s docstring says `t::AbstractVector{Int}`; the method takes `{<:Integer}` and is called with
  `Int32`.
- `decode_heatmap(::MaxDegSolver, …)` says "Returns the node with the highest probability value"
  ([maxdeg.jl:30](src/utils/data/problems/maxdeg.jl#L30)); it returns the top-`k`.
- Every `decode_heatmap` docstring omits the solver argument from its parameter list.
- The `DiffusionModel` call methods are documented as "Constructor"
  ([diffusion_model.jl:17](src/diffusion_model.jl#L17)) — already flagged by `#GR`.
- `RuntimeDataset`'s constructor docstring says `buffer_capacity` "Defaults to 3 times the solver
  capacity" while the signature says `4*max_concurrent_solves`
  ([runtime_dataset.jl:35,44,48](src/utils/data/runtime_dataset.jl#L35-L48)) — and the actual default
  is `3 *`. The docstring signature line is the one that is wrong.
- `AGNN`'s docstring: "Ansiotropic" → "Anisotropic"; it credits Laurent & Bresson 2018 while the
  README says "Joshi/Laurent & Bresson style". Pick one and cite it properly.
- README typos: "isntance" ([README.md:159](README.md#L159)), "pretrained modules"
  ([README.md:195](README.md#L195)), and "EFLOCO, implemented by Wang et al."
  ([README.md:4](README.md#L4)) should be "proposed by".

### D6. Test suite hygiene
- The test item **"Test heatmap_decisiveness bounds and clamping"** is defined twice with near-identical
  bodies ([test_decoding_utils.jl:36](test/cases/test_decoding_utils.jl#L36) and
  [line 46](test/cases/test_decoding_utils.jl#L46)). TestItemRunner filters by name, so the two are
  indistinguishable from the CLI.
- `[cite: 3]` / `[cite: 4]` markers are left in comments throughout `test_diffusion.jl` and
  `test_flow_matching.jl` — evidently paste artefacts.
- Coverage gaps that matter given the bugs above: nothing tests `sample_heatmap`'s trajectory
  (A2 would have been caught by asserting the visited times are monotone and end at the right
  endpoint), nothing tests `corrupt_batch`, nothing tests that MISP decoding actually yields an
  independent set, and nothing exercises the `NodeSelectionProblems.jl` instance generators (A4/A5).
- The two statistical tests allocate `5000×5000` Float32 arrays and several same-sized temporaries —
  ~400 MB of GPU memory per test for a law-of-large-numbers check that `500×500` would settle.

### D7. Configuration
`src/.env` is committed with one user's absolute paths (`/home/e12023149/…`) — already flagged by
`#GR` — and two of its three variables (`INSTANCES_PATH`, `MODELS_PATH`) are read by nothing.

---

## Suggested order of work

1. **A1** (delete the include) — nothing else is verifiable until the package loads.
2. **A2** (DFM time direction) + **B1** (scheduler contract) together — redesign the iterator to
   yield `(τ_from, τ_to)` pairs and make direction explicit. Re-run every EFLOCO experiment.
3. **A3** (loss average), **A10** (label quality logging), **B7** (score by objective, not `|S|`) —
   everything the thesis reports depends on these three.
4. **A4–A8, A12–A15** — small, local, independently testable.
5. **C1/C2/C3** (ATSS) and **C4/C5** (per-step transfers, per-edge time MLP) — the two changes that
   would most change wall-clock time.
6. **B3/B4** (worker lifecycle, deduplicate the Gurobi solvers), **B6** (RNG threading).
7. **D1–D5** — README and docstrings; cheapest fixes, and they are what a reader of the thesis
   meets first.

## Verification (once implementation starts)

- `julia --project=. -e 'using Thesis'` must succeed — the current tree fails at A1.
- Add a scheduler test asserting, for each `(method, scheduler)` pair, that the sequence of times
  `sample_heatmap` actually visits is monotone and starts/ends at the right endpoints. Assert
  `collect(s)` and `[x for x in s]` agree once B1 is fixed.
- Add a training-loop test on a stub `Diffusion` returning a constant loss `c` over `N` batches, and
  assert `loss_history[end] == c` (currently `c·N/(N+1)`).
- Add a MISP decoding test: random graph → arbitrary heatmap → assert the decoded mask is an
  independent set.
- Add `NSPInstance` construction tests for every `graph_type` symbol (`:er, :rr, :ba, :rb,
  :rb_small, :rb_large`) — A4 and A6 both surface immediately.
- `julia --project=. test/runtests.jl` (GPU required).
