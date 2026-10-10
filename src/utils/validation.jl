# src/utils/validation.jl
#
# Solution-quality metrics of a model on a set of labeled instances. Used by the training loop after every epoch
# (next to the validation loss), so the development of the actual solution quality is tracked during training.

export validation_metrics

"""
    validation_metrics(model::DiffusionModel, solver::ProblemSolver, graphs::AbstractVector{<:GNNGraph};
                       batch_size::Int=32)

Samples one heatmap per instance with the inference schedule of `model.method`, decodes it with the decoder of
`solver` and compares the result with the label stored in `ndata.x`. Returns the means over all instances:

    - 'gap': Gap of the decoded solution to the label in percent, 100 · (1 − objective / label objective).
    - 'objective': Objective value of the decoded solution.
    - 'heatmap_mae': Mean absolute error between heatmap and label, per instance up to the complement for
      problems with complement-symmetric labels.
    - 'decisiveness': Mean |2h − 1| of the heatmap (1 = binary heatmap, 0 = undecided).

The network is set to test mode for the sampling and back to training mode afterwards. The function draws from
the global random number generators; the training loop calls it inside its fixed-seed validation step.

Parameters:
    - 'model': The diffusion model to evaluate.
    - 'solver': The problem solver (decoder, objective, label symmetry).
    - 'graphs': The labeled instances.
    - 'batch_size': The number of instances sampled together.
"""
function validation_metrics(model::DiffusionModel, solver::ProblemSolver, graphs::AbstractVector{<:GNNGraph};
                            batch_size::Int=32)
    symmetric = label_symmetric(solver)
    gaps = Float64[]
    objectives = Float64[]
    maes = Float64[]
    decisiveness = Float64[]

    Flux.testmode!(model.gnn)
    try
        for chunk in Iterators.partition(graphs, batch_size)
            g_batch = Flux.batch(collect(chunk)) |> _model_device(model)
            heatmap = Flux.cpu(sample_heatmap(model.method, model, size(g_batch.ndata.x), g_batch))

            offset = 0
            for g in chunk
                h = vec(heatmap[1, (offset + 1):(offset + g.num_nodes)])
                offset += g.num_nodes
                y = vec(g.ndata.x) .> 0.5f0

                objective = evaluate(solver, g, decode_heatmap(solver, g, h))
                reference = evaluate(solver, g, y)
                reference > 0 && push!(gaps, 100 * (1 - objective / reference))
                push!(objectives, objective)

                mae = mean(abs.(h .- y))
                push!(maes, symmetric ? min(mae, mean(abs.(h .- .!y))) : mae)
                push!(decisiveness, heatmap_decisiveness(h)[2])
            end
        end
    finally
        Flux.trainmode!(model.gnn)
    end

    _mean(v) = isempty(v) ? NaN : mean(v)
    return (; gap = _mean(gaps), objective = _mean(objectives), heatmap_mae = _mean(maes),
              decisiveness = _mean(decisiveness))
end
