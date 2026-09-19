# src/diffusion_model.jl
#
# High-level wrapper combining a GNN with a generative method (Diffusion or Flow Matching).
# Manages the primary training loop, batching, loss computation, and inference generation.

using ChainRulesCore: unthunk, AbstractThunk
export DiffusionModel, infer, build_model_input

"""
    DiffusionModel{G<:GNN, M<:Diffusion}
    
Represents a Diffusionmodel consisting of a gnn and a generative method (e.g. Bernoulli Denoising Diffusion or Flow Matching). 
"""
struct DiffusionModel{G<:GNN, M<:Diffusion}
    gnn::G
    method::M
end

Flux.@layer DiffusionModel

_embed_timesteps(t_emb, ::Nothing, x_t) =
    repeat(t_emb, 1, size(x_t, 2))

_embed_timesteps(t_emb, node_indices::AbstractVector, x_t) =
    t_emb[:, node_indices]

"""
    build_model_input(x::AbstractArray, timesteps::AbstractVector{<:Integer}, time_dim::Int, gi)

Single canonical assembly of a GNN forward-pass input: computes the
sinusoidal timestep embedding, broadcasts it from one-per-graph to
one-per-node via `gi` (or repeats it across all nodes if `gi === nothing`,
i.e. a single unbatched graph), and concatenates it in front of the state,
i.e. `x_input = [t_emb; x]`.

This is the single owner of the "pack timestep embedding + state into one
input matrix" contract (B10) — `AGNN`'s forward pass is the corresponding
unpack side (`t_emb = x_input[1:end-fd, :]; x_t = x_input[end-fd+1:end, :]`),
so the two must stay in sync. Previously this packing logic was duplicated
three times: here, inline in `compute_loss_and_gradients`, and as
`_build_model_input` in `discrete_flow_matching.jl`; all three now call this
one function.

Parameters:
    - 'x': the current node-state matrix, shape (x_dim, N).
    - 'timesteps': per-graph (length == num_graphs) or already per-node
      (length == N) integer timesteps.
    - 'time_dim': width of the sinusoidal timestep embedding.
    - 'gi': `g.graph_indicator`, mapping each node to its graph id, or
      `nothing` for a single unbatched graph.
"""
function build_model_input(x::AbstractArray, timesteps::AbstractVector{<:Integer}, time_dim::Int, gi)
    t_emb = get_timestep_embedding(timesteps, time_dim)
    t_emb_nodes = _embed_timesteps(t_emb, gi, x)
    return vcat(t_emb_nodes, x)
end

"""
    (model::DiffusionModel)(x::AbstractArray, t::AbstractVector{Int}, g::GNNGraph)

Defines a forward pass (callable for Flux) of the network.

Parameters:
    - 'x_t': The current state of the solution.
    - 't': The per-graph diffusion/flow timesteps.
    - 'g': A graph(batch).
"""
function (model::DiffusionModel)(x_t::AbstractArray, t::AbstractVector{<:Integer}, g::GNNGraph)
    x_dim    = size(x_t, 1)
    time_dim = model.gnn.config.in_dim - x_dim

    x_input = build_model_input(x_t, t, time_dim, g.graph_indicator)
    return model.gnn(g, x_input)
end

"""
    train_diffusion_model!(model::DiffusionModel, optim, data_loader; epochs::Int=5, val_loader=nothing)

Main training loop for a diffusion model.

Parameters:
    - 'model': The model to train.
    - 'optim': The optimizer.
    - 'data_loader': The dataloader which loads training data lazily.
    - 'epochs': The epochs to train for.
    - 'val_loader': Optional `Flux.DataLoader` over a *fixed*, held-out
      validation set (same `(GNNGraph, name)` shape as `data_loader`). When
      given, validation loss is computed once per epoch with no gradient
      update, so overfitting becomes observable (B7). When `nothing`
      (default), training behaves exactly as before.

Returns `(loss_history, val_loss_history)`; `val_loss_history` is empty when
`val_loader === nothing`. NOTE: this is a breaking change from the previous
single-`Vector` return value — update any external caller accordingly.
"""
function train_diffusion_model!(model::DiffusionModel, optim, data_loader;
                                 epochs::Int=5,
                                 val_loader::Union{Nothing,Flux.DataLoader}=nothing)
    loss_history = Float32[]
    val_loss_history = Float32[]
    @info "Starting training"

    for epoch in 1:epochs
        if epoch > 1 && data_loader.data isa RuntimeDataset
            Thesis.reset!(data_loader.data)
        end

        epoch_loss = 0.0f0
        num_batches = 0
        total_instances = 0

        prefetch_channel = Channel{Tuple{GNNGraph, Vector{String}}}(3, spawn=true) do ch
            for batch in data_loader
                put!(ch, batch)
            end
        end

        t_start_wait = time()

        for (g_batch, names_batch) in prefetch_channel

            wait_time = time() - t_start_wait

            if wait_time > 0.01
                @debug "GPU Starvation Detected! Waited $(round(wait_time, digits=3))s for the next batch."
            end

            batch_size = length(names_batch)
            batch_time = @elapsed begin
                g_batch = g_batch |> Flux.gpu
                x_start = g_batch.ndata.x
                x_t, timesteps, x_0 = corrupt_batch(model.method, x_start, g_batch)

                loss, grads = compute_loss_and_gradients(model.gnn, model.method, x_t, timesteps, g_batch, x_0)

                Flux.update!(optim, model.gnn, grads[1])
            end

            avg_time_per_instance = batch_time / batch_size
            num_batches += 1
            total_instances += batch_size

            @info "Batch $num_batches [$total_instances instances total] | Total: $(round(batch_time, digits=4))s | Instance GPU Time: $(round(avg_time_per_instance, digits=4))s | GPU Wait time (for solver): $(round(wait_time, digits=4))s"

            epoch_loss += loss

            t_start_wait = time()
        end

        avg_loss = epoch_loss / max(1, num_batches)
        push!(loss_history, avg_loss)

        if !isnothing(val_loader)
            val_loss = _validation_loss(model, val_loader)
            push!(val_loss_history, val_loss)
            @info "Epoch $epoch | Avg Train Loss: $avg_loss | Avg Val Loss: $val_loss"
        else
            @info "Epoch $epoch | Avg Loss: $avg_loss"
        end
    end

    return loss_history, val_loss_history
end

"""
    _validation_loss(model::DiffusionModel, val_loader)

Computes the average loss over a held-out validation set without updating
weights. Reuses `compute_loss_and_gradients` (so both `BernoulliDiffusion`
and `DiscreteFlowMatching` are supported through the same generic dispatch
used during training) and simply discards the gradients. This costs one
unused backward pass per validation batch — a deliberate simplicity/DRY
trade-off against writing a third, forward-only loss path; validation sets
are small and this only runs once per epoch.

Parameters:
    - 'model': The model being validated.
    - 'val_loader': A `Flux.DataLoader` over a fixed, held-out set of
      `(GNNGraph, name)` pairs.
"""
function _validation_loss(model::DiffusionModel, val_loader)
    Flux.testmode!(model.gnn)
    total_loss = 0.0f0
    n_batches = 0

    for (g_batch, _) in val_loader
        g_batch = g_batch |> Flux.gpu
        x_start = g_batch.ndata.x
        x_t, timesteps, x_0 = corrupt_batch(model.method, x_start, g_batch)
        loss, _ = compute_loss_and_gradients(model.gnn, model.method, x_t, timesteps, g_batch, x_0)
        total_loss += loss
        n_batches += 1
    end

    Flux.trainmode!(model.gnn)
    return n_batches > 0 ? total_loss / n_batches : NaN32
end

"""
    infer(model::DiffusionModel, g_batch::GNNGraph)

Single-graph(-batch) convenience entry point for inference: moves the batch
to the GPU, samples a heatmap via `sample_heatmap`, and returns it on the
CPU. This is the recommended public API for one-off inference (B11 — it was
previously unexported and unused, so it read as dead code; the more
elaborate evaluation loop in `run_automated_pipeline`, test_utils.jl, calls
`sample_heatmap` directly only because it also needs to interleave batching,
decoding, and result-file writing).

Parameters:
    - 'model': The model which will be infered.
    - 'batch': The batch of graphs which will be passed for inference.
"""
function infer(model::DiffusionModel, batch::GNNGraph)
    g_gpu = batch |> Flux.gpu
    shape = (1, g_gpu.num_nodes)
    heatmap_gpu = Thesis.sample_heatmap(model.method, model, shape, g_gpu)
    return heatmap_gpu |> Flux.cpu
end

"""
Needed to ensure Zygote is not evaluating thunks unnecessarily often. It forces a one-time materialization so the result can be reused.
"""
_mat(g::AbstractThunk)    = unthunk(g)
_mat(g::NamedTuple)       = map(_mat, g)
_mat(g::Tuple)            = map(_mat, g)
_mat(g::AbstractVector)   = map(_mat, g)
_mat(g)                   = g

"""
    compute_loss_and_gradients

Consists of one forward and one backward pass through the network. Returns the gradient and the loss for weight adjustments.

Parameters:
    -'gnn': the GNN of the model
    -'method': the diffusion method
    -'x_t': solution state at timestep t
    -'timesteps': vector containing the diffusion timesteps for each graph in the batch
    -'batch': the batch of graphs, already on the GPU
    -'x_0': the optimal solution for loss computation
"""
function compute_loss_and_gradients(gnn::GNN, method::Diffusion, x_t::AbstractArray, timesteps::AbstractVector, batch::GNNGraph, x_0::AbstractArray)
    x_dim = size(x_t, 1)
    time_dim = gnn.config.in_dim - x_dim
    x_input = build_model_input(x_t, timesteps, time_dim, batch.graph_indicator)

    loss, grads = Flux.withgradient(gnn) do gnn_inner # IMPORTANT TO BOX LIKE THIS FOR ZYGOTE! Otherwise each training call the gradient will be recalculated!
        preds = gnn_inner(batch, x_input)
        calculate_loss(method, preds, x_0)
    end
    return loss, (_mat(grads[1]),)
end

"""
    get_timestep_embedding(timesteps, embedding_dim, max_period=10000)

Generates sinusoidal time embeddings.

Parameters:
    - 'timesteps': The vector of integers (T,)
    - 'embedding_dim': The output dimension (D,)
    - 'max_period': The maximum period used.
"""
function get_timestep_embedding(timesteps::AbstractVector{<:Integer}, embedding_dim::Int, max_period=10000)
    half = div(embedding_dim, 2)
    freqs = exp.(-log(Float32(max_period)) .* Float32.(0:half-1) ./ Float32(half)) |> Flux.gpu
    t = Float32.(timesteps)
    args = reshape(t, :, 1) .* reshape(freqs, 1, :)
    emb = permutedims(hcat(cos.(args), sin.(args)))
    if embedding_dim % 2 == 1
        emb = vcat(emb, CUDA.zeros(Float32, 1, size(emb, 2)))
    end
    return emb
end