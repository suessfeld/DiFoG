# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two things live side by side:

1. **`Thesis.jl`** (`src/`, `test/`) — a Julia package that solves node-selection COPs on graphs
   (MIS, MaxCut, MaxDeg) with generative models: DIFUSCO-style discrete Bernoulli diffusion and
   EFLOCO-style discrete flow matching, both denoised by an anisotropic GNN. Built on `Flux.jl` +
   `GraphNeuralNetworks.jl`.
2. **The thesis text** (`workspace/`) — LaTeX sources for the working paper, chapters, and meeting
   notes. `README.md` documents the package for the thesis author's own use.

## Commands

```bash
# Tests (TestItems.jl / TestItemRunner.jl; runs everything except items tagged :job)
julia --project=. test/runtests.jl

# Single test item — TestItemRunner filters by name
julia --project=. -e 'using TestItemRunner; @run_package_tests filter = ti -> occursin("Cosine", ti.name)'

# Run an experiment pipeline
julia --project=. path/to/your_run_script.jl

# Thesis PDF (biber for \printbibliography)
cd workspace && latexmk -pdf working-paper.tex
```

**Tests require a CUDA GPU.** `q`, `p`, and `AGNN`'s edge-feature init call `CUDA.rand` /
`CUDA.zeros` unconditionally, and the test items build tensors with `|> Flux.gpu`. There is no CPU
fallback path.

## Architecture

Four abstract types compose into a run; everything else is a subtype plugged into one of them.

| Type | File | Contract |
|---|---|---|
| `Diffusion` | `src/diffusion/diffusion.jl` | `corrupt_batch`, `p`, `sample_heatmap`, `calculate_loss` |
| `Scheduler{T}` | `src/diffusion/scheduler.jl` | populate `total_steps`, `inference_steps`, `steps` |
| `GNN` | `src/gnn/gnn.jl` | `(m::YourGNN)(g::GNNGraph, x_input::AbstractMatrix)` → per-node logits |
| `GraphGenerator` / `ProblemSolver` | `src/utils/data/runtime_dataset.jl` | `gen()` → `SimpleGraph`; `solver(g)` → `1×n` label matrix; plus `decode_heatmap(::YourSolver, g, heatmap)` |

`DiffusionModel` ([src/diffusion_model.jl](src/diffusion_model.jl)) is just `(gnn, method)`. It owns
the training loop, the sinusoidal timestep embedding, and `infer`. The method owns the math; the GNN
owns the parameters — `Flux.setup`/`Flux.update!` and `save_model`/`load_model!` all operate on
`model.gnn`, never on `model`.

**Data flow.** `RuntimeDataset` spawns `max_concurrent_solves` background tasks that generate a
random graph, solve it with Gurobi for the ground-truth label, and `put!` it into a bounded
`Channel`. Training pulls batches through a second prefetch channel while the GPU works, and logs a
"GPU Starvation Detected" `@debug` when it waits. Consequences worth knowing:

- `getindex` **ignores its index** and does `take!(buffer)`, so `shuffle=true` on the `DataLoader` is
  a no-op and instances are non-deterministic.
- Workers loop forever and catch-and-resume on error; a persistently failing solver shows up as
  repeated `@error` logs plus a stalled trainer, not a crash.
- Gurobi environments come from a fixed pool of `NUM_GUROBI_LICENSES = 2`
  ([src/utils/data/problems/misp.jl](src/utils/data/problems/misp.jl)), created once in `__init__`.
  Both MISP and MaxCut solvers take from and return to that pool and contain WLS-token recovery
  paths — keep that structure when adding a solver.

**Inference.** `sample_heatmap` iterates the scheduler backwards, calls `p` at each step, and returns
per-node probabilities; `decode_heatmap` dispatches on the *solver* type to turn them into a feasible
solution (greedy independent-set repair for MISP, threshold at 0.5 for MaxCut, top-k for MaxDeg).

**Experiment harness.** `run_automated_pipeline` lives in [test/test_utils.jl](test/test_utils.jl) —
deliberately *outside* the package. It trains, tests, writes per-instance result files and a
ready-to-`\input` LaTeX report. Key behaviour: if `models/model-<run_id>.jld2` already exists,
training is **skipped** and weights are loaded, so reusing a `run_id` silently reuses the old model.
Outputs land in `models/` and `results/` (`results/` is gitignored).

## Conventions and gotchas

- **`src/Thesis.jl` is a flat module with a hand-ordered include list.** Abstract types must be
  included before their implementations, and `gnn_config.jl` / `math_utils.jl` before
  `diffusion_model.jl`. A new file is invisible until added there. `DataUtils` is the only submodule.
- **Scheduler iteration is non-standard.** `iterate` walks `steps` *descending* and starts at
  `length(steps)-1`, so `length(s) == length(s.steps)-1` and `collect(s)` returns the whole `steps`
  vector (it overrides `Base.collect`, it does not iterate). `sample_heatmap` relies on this: it
  seeds `τ_current = total_steps`, consumes the iterator, then does one final step to `0`.
- **Scheduler element type is coupled to the method.** `BernoulliDiffusion` needs `Scheduler{Int}`,
  `DiscreteFlowMatching` needs `Scheduler{Float32}`. `LinearScheduler`/`CosineScheduler` default to
  `Int` for the two-arg form; pass the type explicitly (`CosineScheduler(Float32, 1000, 50)`) or use
  `Base.convert`. `ATSSScheduler` defaults to `Float32` and additionally needs the generator, solver
  and `κ`/`κ̇` because it fits its step sequence by sampling trajectories up front.
- **`in_dim` is time_dim + feature_dim.** `AGNN(config, feature_dim=1)` splits `x_input` as
  `[t_emb; x_t]`, so `GNNConfig(in_dim=33)` means a 32-dimensional sinusoidal time embedding. The
  constructor asserts `in_dim > feature_dim`.
- **Loss/gradient dispatch.** The generic `compute_loss_and_gradients(gnn, method, ...)` in
  `diffusion_model.jl` covers methods that only define `calculate_loss`. `DiscreteFlowMatching`
  overrides it (needs two coupled timesteps for the consistency term) and must write
  `function Thesis.compute_loss_and_gradients(...)` explicitly, since included files evaluate inside
  the module.
- **Zygote:** gradients are taken with `Flux.withgradient(gnn) do gnn_inner ... end` — closing over
  the outer `gnn` instead re-triggers gradient computation every call. `_mat` force-materializes
  thunks once so the result is reusable.
- **Config comes from `src/.env`** (loaded at module load *and* in `__init__`): `LOG_LEVEL`,
  `INSTANCES_PATH`, `MODELS_PATH`, `GRB_LICENSE_FILE`. `set_log_level!` uses
  `Logging.disable_logging`, so `@debug` output requires `LOG_LEVEL=DEBUG` there, not just a logger
  swap.
- **Tests** use `@testmodule` + `@testitem` with `setup=[...]`, one setup module per file under
  `test/cases/`. Tag long-running GPU/cluster work `:job` to keep it out of `runtests.jl`.

## Current inconsistencies

- [src/Thesis.jl:28](src/Thesis.jl#L28) still includes
  `./diffusion/methods/diffusion/gaussian_diffusion.jl`, which was deleted in commit `c965b68`. The
  package does not load until that line goes.
- `README.md`'s module tree and testing section describe the old `src/tests/` layout; the real paths
  are `test/runtests.jl`, `test/test_utils.jl`, `test/cases/`. `.gitignore` still lists
  `src/tests/pipelines` and `src/tests/cases/cluster/*` for the same reason.
- `README.md` lists `GaussianDiffusion` as an implementation of `Diffusion`; it no longer exists.
- `framework.jl` (untracked, repo root) is a standalone draft of a *unified* rewrite — one
  `GenerativeFramework` hierarchy over discrete diffusion / DFM / Gaussian diffusion on a normalized
  `t ∈ [0,1]` timeline. It is not wired into the package and carries `#GR TODO` review markers.
