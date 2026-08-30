# Thesis.jl

A Julia framework for solving Combinatorial Optimization Problems (COPs) like the Maximum Independet Set (MIS) or Maximum Cut (MaxCut) problem on graphs with **generative diffusion models built on Graph Neural Networks**. It implements the discrete
Bernoulli diffusion framework proposed by Sun et al. in **DIFUSCO** and the discrete flow matching framework **EFLOCO**, implemented by Wang et al., both driven by an anisotropic GNN denoiser and a family of pluggable timestep schedulers (linear, cosine, Adaptive Time-step Scheduling Strategy (ATSS)).

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
| `Diffusion` | Defines forward corruption `q` / reverse denoising `p` and the training loss | `BernoulliDiffusion`, `GaussianDiffusion` (unmaintained), `DiscreteFlowMatching` |
| `Scheduler{T}` | Produces the sequence of (integer or continuous) timesteps used during inference | `LinearScheduler`, `CosineScheduler`, `ATSSScheduler` |
| `GNN` | The denoising network, called as `model(g, x_input)` | `AGNN` (Anisotropic GNN, Joshi/Laurent & Bresson style) |
| `GraphGenerator` / `ProblemSolver` | Produce training/test instances at runtime: a random graph and its (near-)optimal label vector | `ERGraphGenerator`, `BasicGraphGenerator` / `MISPSolver`, `MaxCutSolver`, `MaxDegSolver` |


At inference time, `sample_heatmap` iterates the reverse process over the
steps produced by the chosen `Scheduler`, producing a per-node probability
heatmap, which is then greedily decoded into a feasible solution via
`decode_heatmap`.

---

## Module Structure

```
src/
├── Thesis.jl                       # module entry point, include order
├── diffusion_model.jl              # DiffusionModel wrapper, training loop,
├── diffusion/
│   ├── diffusion.jl                # abstract type Diffusion
│   ├── scheduler.jl                # abstract type Scheduler{T} (iterator interface)
│   ├── scheduler/
│   │   ├── linear.jl               # LinearScheduler
│   │   ├── cosine.jl               # CosineScheduler
│   │   └── atss.jl                 # ATSSScheduler (Adaptive Time-Step Scheduling)
│   └── methods/
│       ├── diffusion/
│       │   ├── bernoulli_diffusion.jl   # proposed in DIFUSCO(unmaintained)
│       └── flow_matching/
│           └── discrete_flow_matching.jl # proposed in EFLOCO
├── gnn/
│   ├── gnn.jl                      # abstract type GNN
│   ├── gnn_config.jl               # GNN configuration
│   └── models/
│       └── AGNN.jl                 # Anisotropic GNN used by common literature
├── utils/
│   ├── math_utils.jl
│   ├── model_utils.jl              # save_model / load_model! (JLD2)
│   ├── decoding_utils.jl           # helpers for heatmap decoding
│   ├── logging_utils.jl            # logging utils
│   └── data/
│       ├── data_utils.jl           # normalize_graph
│       ├── runtime_dataset.jl      # RuntimeDataset for training
│       ├── graphs/
│       │   ├── basic.jl            # basic graph implementation
│       │   └── er.jl               # Erdős–Rényi graphs
│       ├── problems/               
│       │   ├── maxcut.jl           # MaxCut problem
│       │   ├── maxdeg.jl           # MaxDeg problem
│       │   ├── misp.jl             # MIS problem             
│       │   └── NodeSelectionProblems.jl # Integration of existing Gurobi Code       
tests/
├── runtests.jl                     # testrunner
├── test_utils.jl                   # test utils (pipeline logic)
└── cases/
    ├── test_data_utils.jl          # testing of data utils
    ├── test_decoding_utils.jl      # testing of decoding utils
    ├── test_diffusion.jl           # testing of diffusion logic (DIFUSCO)
    ├── test_flow_matching.jl       # testing of flow matching logic (EFLOCO)
    ├── test_graph_generator.jl     # testing of the graph generation
    ├── test_math_utils.jl          # testing of used math utils
    └── test_schedulers.jl          # testing of scheduler logic
```

## Usage
The main way this module was used and tested is via pipelines. The `test_utils.jl` provide a set of helpers to set up different configurations efficiently.

### Example Pipeline Usage

```julia
using Pkg
Pkg.activate(".")

using CUDA
using cuDNN
using Flux
using Thesis

# Training and pipeline helpers are outside of the package!
include(joinpath(pkgdir(Thesis), "src", "tests", "test_utils.jl"))

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
    # should be used. Generally those are independent.
    scheduler = CosineScheduler(1000, 50)
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
                                     # the same isntance
    )
end

Base.invokelatest(main)
```

Some noteworthy details for pipeline usage:

- **Naming**: In the provided example the run_id will always be taken from the filename. This results in a clear separation of pipeline runs.
- **Swapping the method**: replace `BernoulliDiffusion(scheduler)` with `DiscreteFlowMatching(scheduler, λ_cons, κ, κ̇)` to switch from diffusion to flow matching; both satisfy the same `Diffusion` interface (`corrupt_batch`, `p`, `sample_heatmap`, `compute_loss_and_gradients`/`calculate_loss`), so the rest of the script is unchanged.
- **Swapping the scheduler**: `LinearScheduler`, `CosineScheduler` take `(total_steps, inference_steps)`; `ATSSScheduler` additionally needs the `generator`, `solver`, and the interpolation function `κ`/`κ̇` to fit an adaptive step sequence up front.
- **Swapping the problem**: pair `MISPSolver(time_limit)` with a MISP-oriented pipeline, or `MaxCutSolver(time_limit)` for Max-Cut — `decode_heatmap` is dispatched on the solver type, so the correct decoding strategy (greedy independent-set repair vs. naive thresholding) is picked up automatically.

### Running a pipeline

```bash
julia --project=. path/to/your_run_script.jl
```

### Outputs

A completed run produces:

```
models/model-<run_id>.jld2                        # trained GNN weights (JLD2, CPU state)
results/report-<run_id>.tex                        # LaTeX tables: config, hyperparams
results/test_instances_<run_id>/
    test_instance_<n>_result.txt                    # test instances (truth + pred.)           
    test_instance_<last>_<r>_result.txt             # repeated inference on last inst.
```

The `.tex` report is written with filled tables.

### Loading a previously trained model

Generally the module will check in the `/models` directory for pretrained modules with the specified run_id. If such model is available, training will be skipped and the loaded model will be used.

However, it can be done manually as well:
```julia
using Thesis, Flux

model = DiffusionModel(AGNN(gnn_config), BernoulliDiffusion(scheduler)) |> Flux.gpu
Thesis.load_model!(model, "models/model-my_run.jld2")

g = ... |> Flux.gpu               # a GNNGraph, e.g. from a GraphGenerator
heatmap = Thesis.infer(model, g)  # runs sample_heatmap and returns it on CPU
```

---

## Extending the framework

Adding a new component follows the same pattern throughout the codebase —
subtype the relevant abstract type, implement its required functions, then
register the file in `src/Thesis.jl`:

- **New diffusion/flow-matching method**: subtype `Diffusion`; implement `corrupt_batch`, `p` (or the method-specific reverse step), `sample_heatmap`, and either `calculate_loss` or a custom `Thesis.compute_loss_and_gradients` overload.
- **New scheduler**: subtype `Scheduler{T}`; provide a constructor that populates `total_steps`, `inference_steps`, and `steps` — iteration, `length`, and `collect` are inherited from the abstract type. Note that any new Scheduler must also follow the `Iterator` design pattern.
- **New GNN architecture**: subtype `GNN`; implement `(model::YourGNN)(g::GNNGraph, x_input::AbstractMatrix)` returning per-node logits, matching the `in_dim`/`hidden_dim`/`output_dim` contract of `GNNConfig`.
- **New problem/solver**: subtype `ProblemSolver`; implement the solver call `(solver::YourSolver)(g::Graphs.SimpleGraph)` returning a `1×n` label matrix, and a `decode_heatmap(::YourSolver, g::GNNGraph, heatmap::AbstractVector)` method for problem specific decoding.
- **New graph generator**: subtype `GraphGenerator`; implement `(gen::YourGenerator)()` returning a `Graphs.SimpleGraph`.
---

## Testing

Tests are written with `TestItems.jl`/`TestItemRunner.jl` under
`src/tests/cases/`.

```bash
julia --project=. src/tests/testrunner.jl
```

This runs all test items except those tagged `:job` (used for long-running
GPU/cluster jobs rather than unit tests).