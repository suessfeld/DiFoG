# DiFoG.jl — Diffusion & Flow Matching for Graphs

A Julia framework for solving Combinatorial Optimization Problems (COPs) like the Maximum Independent Set (MIS) or Maximum Cut (MaxCut) problem on graphs with **generative diffusion models built on Graph Neural Networks**. It implements the discrete
Bernoulli diffusion framework proposed by Sun et al. in **DIFUSCO** and the discrete flow matching framework proposed by Li et al. in **EFLOCO**, both driven by an anisotropic GNN denoiser and a family of pluggable timestep schedulers (linear, cosine, Adaptive Time-step Scheduling Strategy (ATSS)).

The package is built on `Flux.jl` and `GraphNeuralNetworks.jl`, and is
designed around **runtime-generated training data**: graphs and their
(near-)optimal solutions are produced lazily and concurrently on background
threads while the GPU trains on previous batches.

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Module Structure](#module-structure)
- [Usage](#usage)
  - [Example Pipeline Usage](#example-pipeline-usage)
  - [Outputs](#outputs)
  - [Loading a previously trained model](#loading-a-previously-trained-model)
- [Extending the framework](#extending-the-framework)
- [Testing](#testing)

---

## Core Concepts

The package is organized around four abstract type hierarchies that plug
into each other:

| Abstraction | Role | Implementations |
|---|---|---|
| `Diffusion` | Defines forward corruption `q` / reverse denoising `p` and the training loss | `BernoulliDiffusion`, `DiscreteFlowMatching` |
| `Scheduler{T}` | Produces the sequence of (integer or continuous) timesteps used during inference | `LinearScheduler`, `CosineScheduler`, `ATSSScheduler` |
| `GNN` | The denoising network, called as `model(g, x_input)` | `AGNN` (Anisotropic GNN, Joshi/Laurent & Bresson style) |
| `GraphGenerator` / `ProblemSolver` | Produce training/test instances at runtime: a random graph and its (near-)optimal label vector | `ERGraphGenerator`, `BasicGraphGenerator` / `MISPSolver`, `MaxCutSolver`, `MaxDegSolver` |

Both `Diffusion` implementations share the same coupling to `Scheduler{T}`'s
numeric type parameter: `BernoulliDiffusion` requires a `Scheduler{Int}`
(the reverse process walks a discrete step index), while
`DiscreteFlowMatching` requires a `Scheduler{Float32}` (it interpolates over
a continuous time `t ∈ [0,1]`). `LinearScheduler`/`CosineScheduler` default
to `Scheduler{Int}` unless you explicitly construct them as
`LinearScheduler(Float32, total_steps, inference_steps)`, so switching a
pipeline between methods generally also means switching (or explicitly
typing) the scheduler — see [Swapping the method](#example-pipeline-usage)
below.

At inference time, `sample_heatmap` iterates the reverse process over the
steps produced by the chosen `Scheduler`, producing a per-node probability
heatmap, which is then greedily decoded into a feasible solution via
`decode_heatmap`.

---

## Module Structure
```
src/
├── DiFoG.jl # module entry point, include order
├── diffusion_model.jl # DiffusionModel wrapper, training loop, build_model_input
├── diffusion/
│ ├── diffusion.jl # abstract type Diffusion
│ ├── scheduler.jl # abstract type Scheduler{T} (iterator interface)
│ ├── scheduler/
│ │ ├── linear.jl # LinearScheduler
│ │ ├── cosine.jl # CosineScheduler
│ │ └── atss.jl # ATSSScheduler (Adaptive Time-Step Scheduling)
│ └── methods/
│ ├── diffusion/
│ │ └── bernoulli_diffusion.jl # proposed in DIFUSCO
│ └── flow_matching/
│ └── discrete_flow_matching.jl # proposed in EFLOCO
├── gnn/
│ ├── gnn.jl # abstract type GNN
│ ├── gnn_config.jl # GNN configuration
│ └── models/
│ └── AGNN.jl # Anisotropic GNN used by common literature
├── graphs/
│ ├── basic.jl # BasicGraphGenerator
│ └── er.jl # ERGraphGenerator (Erdős–Rényi)
├── problems/
│ ├── maxcut.jl # MaxCut problem (MaxCutSolver)
│ ├── maxdeg.jl # MaxDeg problem (MaxDegSolver)
│ ├── misp.jl # MIS problem (MISPSolver)
│ └── NodeSelectionProblems.jl # shared NSP/ILP machinery (Gurobi)
└── utils/
├── model_utils.jl # save_model / load_model! (JLD2)
├── decoding_utils.jl # heatmap decoding/evaluation helpers
├── logging_utils.jl # logging utils
├── tracking.jl # experiment tracking (AbstractTracker, NoTracker, WandbTracker), loss/metric history writers
├── validation.jl # validation_metrics: solution quality on labeled instances
└── data/
├── data_utils.jl # normalize_graph
└── runtime_dataset.jl # RuntimeDataset for training

test/ # TestItem tests only, own project (test/Project.toml)
├── Project.toml
├── runtests.jl # test runner (TestItemRunner)
└── cases/
│   ├── test_data_utils.jl
│   ├── test_decoding_utils.jl
│   ├── test_diffusion.jl # DIFUSCO / BernoulliDiffusion tests
│   ├── test_flow_matching.jl # EFLOCO / DiscreteFlowMatching tests
│   ├── test_graph_generators.jl
│   ├── test_schedulers.jl
└── └── test_tracking.jl # tracker interface, WandbTracker (offline run), validation metrics, metric histories

evaluation/ # use of the package: experiments of the thesis, own project (evaluation/Project.toml)
├── Project.toml
├── pipeline.jl # pipeline helper (run_automated_pipeline)
├── create_dataset.jl # labels a seeded dataset
├── evaluate_50_100.jl # experiment grid for one instance size (also 300_400, 700_800)
├── helpers/ # Gurobi convergence study, LNS calibration, step-rule comparison
└── slurm/ # batch scripts for the cluster

CondaPkg.toml # Python version for the client of Wandb.jl
```
## Usage
The main way this module was used and tested is via pipelines. `evaluation/pipeline.jl` provides a set of helpers to set up different configurations efficiently — most importantly `run_automated_pipeline`, which handles training, evaluation, and report generation end to end.

### Example Pipeline Usage

```julia
using Pkg
Pkg.activate("evaluation")

using CUDA
using cuDNN
using Flux
using DiFoG

# Training and pipeline helpers are outside of the package!
include(joinpath(pkgdir(DiFoG), "evaluation", "pipeline.jl"))

function main()
    # `run_id` is derived from the script's own filename.
    current_file = basename(@__FILE__)
    run_id = replace(current_file, ".jl" => "")

    # 1) Data: For Training and Testing both graph type and problem solver has to be
    # defined.
    generator = ERGraphGenerator(min_nodes=300, max_nodes=400, p=0.15)
    solver = MaxCutSolver(time_limit=0.2)   # Gurobi time limit for training due to computational limitation

    # 2) GNN: Defines the GNN configuration used. Both DIFUSCO and EFLOCO use 12
    # layers with a hidden dim of 256.
    gnn_config = GNNConfig(
        num_layers=12,
        in_dim=33,
        hidden_dim=256,
        output_dim=1
    )
    gnn = AGNN(gnn_config)

    # 3) Method/Scheduler: Define what denoising/diffusion method and scheduler
    # should be used.
    scheduler = CosineScheduler(1000, 50)   # defaults to Scheduler{Int}
    method = BernoulliDiffusion(scheduler)

    # 4) Compose and move the whole model to the GPU.
    model = DiffusionModel(gnn, method) |> Flux.gpu

    # 5) Run pipeline
    run_automated_pipeline(;
        run_id = run_id,
        model = model,
        generator = generator,
        solver = solver,
        num_train = 1000,
        num_test = 100,
        epochs = 10,
        batch_size = 2,
        learning_rate = 0.002,
        num_repeat_inference = 100   # repeated re-inference on last test
                                     # instance, to measure sampling variation on
                                     # the same instance
    )
end

Base.invokelatest(main)
```

Some noteworthy details for pipeline usage:

- **Naming**: In the provided example the run_id will always be taken from the filename. This results in a clear separation of pipeline runs.
- **Existing models are loaded automatically**: `run_automated_pipeline` checks `models/model-<run_id>.jld2` before training; if a matching file exists, training is skipped entirely and the saved weights are loaded instead. Delete (or rename) the file to force a fresh training run under the same `run_id`.
- **Swapping the method**: replace `BernoulliDiffusion(scheduler)` with `DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇)` to switch from diffusion to flow matching. Both satisfy the same broad `Diffusion` interface (`corrupt_batch`, `p`, `sample_heatmap`, `compute_loss_and_gradients`/`calculate_loss`), **but the rest of the script is not fully interchangeable**:
  - `DiscreteFlowMatching` requires a `Scheduler{Float32}`; construct the scheduler as `CosineScheduler(Float32, total_steps, inference_steps)` (or the `Int`-based default won't satisfy the constructor).
  - `DiscreteFlowMatching` additionally needs `λ_cons` (consistency-loss weight) and the interpolation function `κ` plus its derivative `κ̇`, e.g. `κ = t -> t`, `κ̇ = t -> 1.0f0`.
  - The two methods' `p` step functions differ in signature (`p(..., t::Int, s::Int, g)` for Bernoulli vs. `p(..., τ::Real, τ_next::Real, g)` for flow matching) — this only matters if you're calling `p` directly rather than through `sample_heatmap`.
- **Swapping the scheduler**: `LinearScheduler`, `CosineScheduler` take `(total_steps, inference_steps)`, or `(T, total_steps, inference_steps)` where `T` is `Int` or `Float32` to pick the eltype explicitly; `ATSSScheduler` additionally needs the `generator`, `solver`, and the interpolation function `κ`/`κ̇` to fit an adaptive step sequence up front.
- **Swapping the problem**: pair `MISPSolver(time_limit)` with a MISP-oriented pipeline, or `MaxCutSolver(time_limit)` for Max-Cut — `decode_heatmap` is dispatched on the solver type, so the correct decoding strategy (greedy independent-set repair vs. naive thresholding) is picked up automatically.
- **Validation**: by default, `run_automated_pipeline` builds a small held-out validation set (`num_val`, default `max(20, num_train÷20)`). After every epoch it reports the validation loss and the solution quality on this set (`validation_metrics`): the instances are sampled with the inference schedule of the model, decoded, and compared with their labels (mean gap, objective, heatmap error, decisiveness). Pass `num_val = 0` to skip this.

### Running a pipeline

```bash
julia --project=evaluation path/to/your_run_script.jl
```

Everything that uses the package (experiment scripts, pipeline helper) lives in `evaluation/`, with its own project (`evaluation/Project.toml`). Like `test/`, it is part of the workspace of the package and therefore shares its manifest. Set it up once with

```bash
julia --project=evaluation -e 'using Pkg; Pkg.instantiate()'
```

### Experiment tracking

The training loop reports its metrics to a tracker (`AbstractTracker`): the training loss of every batch, and after every epoch the validation loss and the solution-quality metrics. The default `NoTracker` discards them. `WandbTracker(; project, name, config, kwargs...)` reports them to Weights & Biases and creates one W&B run. Wandb.jl is a dependency of DiFoG and installs the Python client when it is first loaded (CondaPkg, into `.CondaPkg/` next to the active project; the Python version is set in `CondaPkg.toml`). Authenticate once with `WANDB_API_KEY` (or `wandb login`); `WANDB_MODE=offline` writes the runs to `wandb/` for a later `wandb sync`.

The loss and metric histories are also written as plain text (`graphs/` of a result root), from which plots can be made with any tool.

```julia
using DiFoG

tracker = DiFoG.WandbTracker(; project="DiFoG", name="misp_diffusion", config=Dict("epochs" => "20"))
history = DiFoG.train_diffusion_model!(model, optim, train_loader; epochs=20, val_loader, solver,
                                        val_graphs, tracker)
DiFoG.finish!(tracker)
```

`run_automated_pipeline` takes a function `make_tracker = (name, config) -> tracker` instead and calls it only when a model is actually trained. The evaluation scripts are controlled by environment variables:

| Variable | Effect |
|---|---|
| `DIFOG_WANDB=true` | track every training run in W&B (project `WANDB_PROJECT`, default `DiFoG`) |
| `DIFOG_RESULT_ROOT=<dir>` | write to another result directory (its `seeds/` must contain the labeled datasets) |
| `DIFOG_TRAIN_ONLY=true` | train the models only, without the test evaluation |

```bash
DIFOG_WANDB=true DIFOG_TRAIN_ONLY=true DIFOG_RESULT_ROOT=results/results_50_100_tracked \
    julia evaluation/evaluate_50_100.jl
```

### Outputs
```
A completed run produces:
models/model-<run_id>.jld2 # trained GNN weights (JLD2, CPU state)
graphs/loss-<run_id>.txt # training loss per batch, validation loss per epoch
graphs/metrics-<run_id>.csv # all metrics per epoch (losses, validation gap, objective, heatmap error)
results/report-<run_id>.tex # latex report
results/test_instances_<run_id>/test_instance_<n>result.txt # test instances (truth + pred.)
test_instance<last>_<r>_result.txt # repeated inference on last inst.
```

### Loading a previously trained model

By default, `run_automated_pipeline` checks the `models/` directory for a
pretrained model matching the given `run_id`. If found, training is
skipped and the saved weights are loaded instead.

It can also be done manually:
```julia
using DiFoG, Flux

model = DiffusionModel(AGNN(gnn_config), BernoulliDiffusion(scheduler)) |> Flux.gpu
DiFoG.load_model!(model, "models/model-my_run.jld2")

g = ... |> Flux.gpu               # a GNNGraph, e.g. from a GraphGenerator
heatmap = DiFoG.infer(model, g)  # runs sample_heatmap and returns it on CPU
```

---

## Extending the framework

Adding a new component follows the same pattern throughout the codebase —
subtype the relevant abstract type, implement its required functions, then
register the file in `src/DiFoG.jl`:

- **New diffusion/flow-matching method**: subtype `Diffusion`; implement `corrupt_batch`, `p` (or the method-specific reverse step), `sample_heatmap`, and either `calculate_loss` or a custom `DiFoG.compute_loss_and_gradients` overload.
- **New scheduler**: subtype `Scheduler{T}`; provide a constructor that populates `total_steps`, `inference_steps`, and `steps` — iteration, `length`, and `collect` are inherited from the abstract type. Note that any new Scheduler must also follow the `Iterator` design pattern (ascending by default; use `Iterators.reverse` for descending traversal).
- **New GNN architecture**: subtype `GNN`; implement `(model::YourGNN)(g::GNNGraph, x_input::AbstractMatrix)` returning per-node logits, matching the `in_dim`/`hidden_dim`/`output_dim` contract of `GNNConfig`.
- **New problem/solver**: subtype `ProblemSolver`; implement the solver call `(solver::YourSolver)(g::Graphs.SimpleGraph)` returning a `1×n` label matrix, and `decode_heatmap`/`evaluate` methods dispatched on your solver type for problem-specific decoding and scoring.
- **New graph generator**: subtype `GraphGenerator`; implement `(gen::YourGenerator)()` returning a `Graphs.SimpleGraph`.

---

## Testing

Tests are written with `TestItems.jl`/`TestItemRunner.jl` under
`test/cases/`. The directory contains nothing else and has its own project
(`test/Project.toml`), which is part of the workspace of the package.

```bash
julia --project=test test/runtests.jl
```

This runs all test items except those tagged `:job` (used for long-running
GPU/cluster jobs rather than unit tests). A GPU is required, since most
tests exercise CUDA-resident code paths directly.