# src/diffusion_model.jl
#
# High-level wrapper combining a GNN with a generative method (Diffusion or Flow Matching).
# Manages the primary training loop, batching, loss computation, and inference generation.

using ChainRulesCore: unthunk, AbstractThunk
using NNlib
using Random
export DiffusionModel, infer, compute_loss, compute_loss_and_gradients

"""
    DiffusionModel{G<:GNN, M<:Diffusion}

A diffusion model combining a Graph Neural Network (GNN) and a generative method.

Fields:
    - 'gnn': The GNN backbone model.
    - 'method': The generative diffusion or flow matching method.
"""
struct DiffusionModel{G<:GNN, M<:Diffusion}
    gnn::G
    method::M
end

Flux.@layer DiffusionModel

"""
    _model_device(model::DiffusionModel)

Returns the device transfer function matching the model's parameters: `Flux.cpu` for a model on the CPU,
`Flux.gpu` otherwise. Data is moved with it, so a CPU model also works on a machine with a GPU.

Parameters:
    - 'model': The diffusion model.
"""
_model_device(model::DiffusionModel) = first(Flux.trainables(model.gnn)) isa Array ? Flux.cpu : Flux.gpu

"""
    _nodewise(v::AbstractVector, gi, N::Int)

Broadcasts a per-graph vector to one entry per node using the graph indicator, or repeats it for a single unbatched graph.

Parameters:
    - 'v': A per-graph vector to broadcast.
    - 'gi': The graph indicator mapping nodes to graphs (or nothing for unbatched).
    - 'N': The number of nodes or repetitions for unbatched graphs.
"""
function _nodewise(v::AbstractVector, ::Nothing, N::Int)
    @assert length(v) == 1 "unbatched graph expects exactly one per-graph value, got $(length(v))"
    return repeat(v, N)
end
_nodewise(v::AbstractVector, gi::AbstractVector, N::Int) = v[gi]

"""
    _node_row(v::AbstractVector, gi, N::Int)

Reshapes a broadcasted per-graph vector into a row matrix to broadcast against node-state matrices.

Parameters:
    - 'v': A per-graph vector to broadcast.
    - 'gi': The graph indicator mapping nodes to graphs.
    - 'N': The total number of nodes.
"""
_node_row(v::AbstractVector, gi, N::Int) = reshape(_nodewise(v, gi, N), 1, :)

"""
    _device_anchor(g::GNNGraph)

Retrieves an array from the graph that lives on the target device, used as a template for allocating new arrays.

Parameters:
    - 'g': The input graph.
"""
_device_anchor(g::GNNGraph) = edge_index(g)[1]

"""
    _random_binary(g::GNNGraph, shape::Tuple)

Generates a uniform binary source state directly on the same device as the input graph.

Parameters:
    - 'g': The reference graph determining the target device.
    - 'shape': The dimensions of the generated binary state.
"""
_random_binary(g::GNNGraph, shape::Tuple) =
    Float32.(rand!(similar(_device_anchor(g), Float32, shape)) .> 0.5f0)

"""
    _arange_like(x::AbstractArray, n::Int)

Creates a sequential array directly on the device of the reference array, preventing host-to-device copy overhead.

Parameters:
    - 'x': The reference array determining the target device.
    - 'n': The number of sequential elements to generate.
"""
_arange_like(x::AbstractArray, n::Int) = cumsum(fill!(similar(x, Float32, n), 1f0)) .- 1f0

"""
    get_timestep_embedding(timesteps::AbstractVector{<:Integer}, embedding_dim::Int, max_period=10000)

Generates a sinusoidal timestep embedding matrix directly on the target device.

Parameters:
    - 'timesteps': A vector of timesteps to embed.
    - 'embedding_dim': The dimensionality of the resulting embeddings.
    - 'max_period': The maximum period for the sinusoidal frequencies.
"""
function get_timestep_embedding(timesteps::AbstractVector{<:Integer}, embedding_dim::Int, max_period=10000)
    half = div(embedding_dim, 2)
    t = reshape(Float32.(timesteps), 1, :)                                          # (1, n)
    freqs = exp.(-log(Float32(max_period)) .* _arange_like(t, half) ./ Float32(half))  # (half,)
    args = freqs .* t                                                               # (half, n)
    emb = vcat(cos.(args), sin.(args))
    if isodd(embedding_dim)
        emb = vcat(emb, fill!(similar(emb, 1, size(emb, 2)), 0f0))
    end
    return emb
end

"""
    _time_embedding(gnn::GNN, timesteps::AbstractVector{<:Integer}, g::GNNGraph)

Generates per-graph timestep embeddings based on the GNN's configuration.

Parameters:
    - 'gnn': The Graph Neural Network containing the configuration.
    - 'timesteps': A vector providing one timestep per graph.
    - 'g': The batch of graphs.
"""
function _time_embedding(gnn::GNN, timesteps::AbstractVector{<:Integer}, g::GNNGraph)
    @assert length(timesteps) == g.num_graphs "expected one timestep per graph ($(g.num_graphs)), got $(length(timesteps))"
    return get_timestep_embedding(timesteps, gnn.config.time_dim)
end

"""
    (model::DiffusionModel)(x_t::AbstractMatrix, t::AbstractVector{<:Integer}, g::GNNGraph)

Executes the forward pass of the diffusion model to predict logits for the target state.

Parameters:
    - 'model': The DiffusionModel instance.
    - 'x_t': The node state matrix at the current timestep.
    - 't': A vector of per-graph timesteps.
    - 'g': The batch of graphs.
"""
function (model::DiffusionModel)(x_t::AbstractMatrix, t::AbstractVector{<:Integer}, g::GNNGraph)
    return model.gnn(g, x_t, _time_embedding(model.gnn, t, g))
end

"""
    _node_bce(logits::AbstractArray, y::AbstractArray, gi, num_graphs::Int, symmetric::Bool)

Calculates the mean binary cross-entropy over all nodes, optionally applying flip-invariant symmetry per graph.

Parameters:
    - 'logits': The predicted logits.
    - 'y': The target binary labels.
    - 'gi': The graph indicator mapping nodes to graphs.
    - 'num_graphs': The total number of graphs in the batch.
    - 'symmetric': Boolean flag to evaluate loss up to complement symmetry.
"""
function _node_bce(logits::AbstractArray, y::AbstractArray, gi, num_graphs::Int, symmetric::Bool)
    symmetric || return Flux.logitbinarycrossentropy(logits, y)
    l_a = Flux.logitbinarycrossentropy(logits, y; agg=identity)
    l_b = Flux.logitbinarycrossentropy(logits, 1f0 .- y; agg=identity)
    if isnothing(gi)
        return min(sum(l_a), sum(l_b)) / length(y)
    end
    s_a = NNlib.scatter(+, vec(l_a), gi; dstsize=(num_graphs,))
    s_b = NNlib.scatter(+, vec(l_b), gi; dstsize=(num_graphs,))
    return sum(min.(s_a, s_b)) / length(y)
end

"""
    _prepare_loss

Method-specific interface to compute all non-differentiated components (embeddings, schedules, targets) prior to loss calculation.
"""
function _prepare_loss end

"""
    _loss_terms

Method-specific interface to compute the differentiated components of the loss.
"""
function _loss_terms end

"""
    compute_loss(gnn::GNN, method::Diffusion, x_t::AbstractArray, timesteps::AbstractVector, batch::GNNGraph, x_target::AbstractArray, aux=nothing; symmetric::Bool=false)

Performs a forward-only loss computation without tracking gradients, typically used for validation.

Parameters:
    - 'gnn': The GNN backbone model.
    - 'method': The generative diffusion method.
    - 'x_t': The node state matrix at the current timestep.
    - 'timesteps': A vector of per-graph timesteps.
    - 'batch': The batch of graphs.
    - 'x_target': The target solution state.
    - 'aux': Optional method-specific auxiliary data.
    - 'symmetric': Boolean flag to apply flip-invariant loss.
"""
function compute_loss(gnn::GNN, method::Diffusion, x_t::AbstractArray, timesteps::AbstractVector,
                      batch::GNNGraph, x_target::AbstractArray, aux=nothing; symmetric::Bool=false)
    prep = _prepare_loss(gnn, method, x_t, timesteps, batch, x_target, aux, symmetric)
    return _loss_terms(gnn, method, prep)
end

"""
    _mat(g)

Forces a single materialization of thunks to prevent unnecessary evaluations by Zygote, returning numeric arrays unchanged.

Parameters:
    - 'g': The object or collection of objects to materialize.
"""
_mat(g::AbstractThunk)    = unthunk(g)
_mat(g::NamedTuple)       = map(_mat, g)
_mat(g::Tuple)            = map(_mat, g)
_mat(g::AbstractVector)   = eltype(g) <: Number ? g : map(_mat, g)
_mat(g)                   = g

"""
    compute_loss_and_gradients(gnn::GNN, method::Diffusion, x_t::AbstractArray, timesteps::AbstractVector, batch::GNNGraph, x_target::AbstractArray, aux=nothing; symmetric::Bool=false)

Performs a combined forward and backward pass, returning the loss components and computed gradients.

Parameters:
    - 'gnn': The GNN backbone model.
    - 'method': The generative diffusion method.
    - 'x_t': The state at time t.
    - 'timesteps': A vector of per-graph timesteps.
    - 'batch': The batch of graphs.
    - 'x_target': The target solution state.
    - 'aux': Optional method-specific auxiliary data.
    - 'symmetric': Boolean flag to apply flip-invariant loss.
"""
function compute_loss_and_gradients(gnn::GNN, method::Diffusion, x_t::AbstractArray, timesteps::AbstractVector,
                                    batch::GNNGraph, x_target::AbstractArray, aux=nothing; symmetric::Bool=false)
    prep = _prepare_loss(gnn, method, x_t, timesteps, batch, x_target, aux, symmetric)
    loss_parts, grads = Flux.withgradient(gnn) do gnn_inner # IMPORTANT TO BOX LIKE THIS FOR ZYGOTE!
        _loss_terms(gnn_inner, method, prep)                # NamedTuple: gradient of the first field (`loss`)
    end
    return loss_parts, (_mat(grads[1]),)
end

_format_parts(parts::NamedTuple) =
    join(["$(k)=$(round(Float64(v), digits=4))" for (k, v) in pairs(parts) if k != :loss], " | ")

_sync_device() = (CUDA.functional() && CUDA.synchronize(); nothing)

"""
    train_diffusion_model!(model::DiffusionModel, optim, data_loader; epochs::Int=5, val_loader::Union{Nothing,Flux.DataLoader}=nothing, symmetric_labels::Bool=false)

Executes the main training loop for a diffusion model, tracking loss history over batches and epochs.

Parameters:
    - 'model': The diffusion model to train.
    - 'optim': The optimizer state.
    - 'data_loader': The dataloader providing training data batches.
    - 'epochs': The total number of epochs to train for.
    - 'val_loader': An optional dataloader for held-out validation.
    - 'symmetric_labels': Boolean flag to apply flip-invariant loss for complement-symmetric labels.
"""
function train_diffusion_model!(model::DiffusionModel, optim, data_loader;
                                 epochs::Int=5,
                                 val_loader::Union{Nothing,Flux.DataLoader}=nothing,
                                 symmetric_labels::Bool=false)
    loss_history = Float32[]
    val_loss_history = Float32[]
    batch_loss_history = Float32[]
    val_loss_batches = Int[]
    global_batch_num = 0
    @info "Starting training (symmetric_labels=$symmetric_labels)"

    for epoch in 1:epochs
        if epoch > 1 && data_loader isa Flux.DataLoader && data_loader.data isa RuntimeDataset
            DiFoG.reset!(data_loader.data)
        end

        epoch_loss = 0.0f0
        num_batches = 0
        total_instances = 0

        prefetch_channel = Channel{Tuple{GNNGraph, Vector{String}}}(3, spawn=true) do ch
            for batch in data_loader
                put!(ch, batch)
            end
        end

        for (g_batch, names_batch) in prefetch_channel

            batch_size = length(names_batch)
            loss_parts = nothing
            batch_time = @elapsed begin
                g_batch = g_batch |> _model_device(model)
                x_start = g_batch.ndata.x
                x_t, timesteps, x_target, aux = corrupt_batch(model.method, x_start, g_batch)

                loss_parts, grads = compute_loss_and_gradients(model.gnn, model.method, x_t, timesteps,
                                                               g_batch, x_target, aux; symmetric=symmetric_labels)

                Flux.update!(optim, model.gnn, grads[1])
                _sync_device()   # P7: include backward/optimizer work in this batch's timing
            end
            loss = Float32(loss_parts.loss)

            avg_time_per_instance = batch_time / batch_size
            num_batches += 1
            global_batch_num += 1
            total_instances += batch_size

            parts_str = _format_parts(loss_parts)
            @info "Batch $num_batches [$total_instances instances total] | Total: $(round(batch_time, digits=4))s | Instance GPU Time: $(round(avg_time_per_instance, digits=4))s | Loss: $(round(loss, digits=4))" * (isempty(parts_str) ? "" : " | " * parts_str)

            epoch_loss += loss
            push!(batch_loss_history, loss)
        end

        avg_loss = epoch_loss / max(1, num_batches)
        push!(loss_history, avg_loss)

        if !isnothing(val_loader)
            val_loss = _validation_loss(model, val_loader; symmetric_labels=symmetric_labels)
            push!(val_loss_history, val_loss)
            push!(val_loss_batches, global_batch_num)
            @info "Epoch $epoch | Avg Train Loss: $avg_loss | Avg Val Loss: $val_loss"
        else
            @info "Epoch $epoch | Avg Loss: $avg_loss"
        end
    end

    return loss_history, val_loss_history, batch_loss_history, val_loss_batches
end

const VALIDATION_SEED = 0x5eed_7a1d

"""
    _with_fixed_rng(f, seed)

Executes a function using a fixed random seed for both CPU and GPU (if available), subsequently restoring the CPU RNG state to maintain reproducibility.

Parameters:
    - 'f': The zero-argument function to execute.
    - 'seed': The random seed to apply.
"""
function _with_fixed_rng(f, seed)
    cpu_state = copy(Random.default_rng())
    Random.seed!(seed)
    CUDA.functional() && CUDA.seed!(seed)
    try
        return f()
    finally
        copy!(Random.default_rng(), cpu_state)
        CUDA.functional() && CUDA.seed!(rand(Random.default_rng(), UInt32))
    end
end

"""
    _validation_loss(model::DiffusionModel, val_loader; symmetric_labels::Bool=false)

Computes the average forward-only loss over a validation set using a fixed random seed for consistent evaluation.

Parameters:
    - 'model': The diffusion model to evaluate.
    - 'val_loader': The dataloader providing validation data.
    - 'symmetric_labels': Boolean flag to apply flip-invariant loss.
"""
function _validation_loss(model::DiffusionModel, val_loader; symmetric_labels::Bool=false)
    return _with_fixed_rng(VALIDATION_SEED) do
        _validation_loss_unseeded(model, val_loader, symmetric_labels)
    end
end

function _validation_loss_unseeded(model::DiffusionModel, val_loader, symmetric_labels::Bool)
    Flux.testmode!(model.gnn)
    total_loss = 0.0f0
    n_batches = 0

    for (g_batch, _) in val_loader
        g_batch = g_batch |> _model_device(model)
        x_start = g_batch.ndata.x
        x_t, timesteps, x_target, aux = corrupt_batch(model.method, x_start, g_batch)
        parts = compute_loss(model.gnn, model.method, x_t, timesteps, g_batch, x_target, aux;
                             symmetric=symmetric_labels)
        total_loss += Float32(parts.loss)
        n_batches += 1
    end

    Flux.trainmode!(model.gnn)
    return n_batches > 0 ? total_loss / n_batches : NaN32
end

"""
    infer(model::DiffusionModel, batch::GNNGraph)

Performs inference on a batch of graphs, sampling a heatmap and transferring the result back to the CPU.

Parameters:
    - 'model': The trained diffusion model.
    - 'batch': The batch of graphs to process.
"""
function infer(model::DiffusionModel, batch::GNNGraph)
    g_gpu = batch |> _model_device(model)
    shape = (model.gnn.config.in_dim, g_gpu.num_nodes)   # M7
    heatmap_gpu = DiFoG.sample_heatmap(model.method, model, shape, g_gpu)
    return heatmap_gpu |> Flux.cpu
end