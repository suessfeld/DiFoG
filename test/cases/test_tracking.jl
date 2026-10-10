using TestItems
using DiFoG

@testmodule TestTracking begin
    using Test
    using DiFoG
    using Flux
    using GraphNeuralNetworks

    export RecordingTracker, toy_instances

    # Tracker that keeps everything it is given, to inspect what the training loop reports.
    struct RecordingTracker <: DiFoG.AbstractTracker
        logs::Vector{Tuple{Int,Dict{String,Float64}}}
        finished::Base.RefValue{Bool}
    end
    RecordingTracker() = RecordingTracker(Tuple{Int,Dict{String,Float64}}[], Ref(false))
    DiFoG.log_metrics!(t::RecordingTracker, metrics::AbstractDict; step::Integer) =
        (push!(t.logs, (Int(step), Dict{String,Float64}(metrics))); nothing)
    DiFoG.finish!(t::RecordingTracker) = (t.finished[] = true; nothing)

    # Labelled 4-cycles and triangles; the labels are maximum independent sets and also maximum cuts.
    function toy_instances(n::Int)
        cycle() = GNNGraph(Int32[1, 2, 3, 4, 2, 3, 4, 1], Int32[2, 3, 4, 1, 1, 2, 3, 4]; num_nodes=4,
                           ndata = (; x = Float32[1 0 1 0]))
        triangle() = GNNGraph(Int32[1, 2, 3, 2, 3, 1], Int32[2, 3, 1, 1, 2, 3]; num_nodes=3,
                              ndata = (; x = Float32[1 0 0]))
        return [(isodd(i) ? cycle() : triangle(), "inst_$i") for i in 1:n]
    end
end

@testitem "Test validation_metrics on labelled instances" setup=[TestTracking] begin
    using Flux
    using Random
    Random.seed!(1)
    cfg = GNNConfig(num_layers=2, in_dim=1, time_dim=8, hidden_dim=16, output_dim=1)
    graphs = first.(toy_instances(6))

    for solver in (MISPSolver(time_limit=1.0), MaxCutSolver(time_limit=1.0)),
        method in (BernoulliDiffusion(LinearScheduler(100, 3)), RectifiedFlow(LinearScheduler(100, 3)))
        model = DiffusionModel(AGNN(cfg), method)
        m = validation_metrics(model, solver, graphs; batch_size=4)
        @test keys(m) == (:gap, :objective, :heatmap_mae, :decisiveness)
        @test all(isfinite, values(m))
        @test m.gap ≤ 100 && m.objective ≥ 0
        @test 0 ≤ m.heatmap_mae ≤ 1 && 0 ≤ m.decisiveness ≤ 1
        # complement-symmetric labels: the error is measured up to the complement
        DiFoG.label_symmetric(solver) && @test m.heatmap_mae ≤ 0.5
    end
end

@testitem "Test training loop reports metrics and is unchanged by the validation metrics" setup=[TestTracking] begin
    using Flux
    using Random
    cfg = GNNConfig(num_layers=2, in_dim=1, time_dim=8, hidden_dim=16, output_dim=1)
    solver = MISPSolver(time_limit=1.0)
    instances = toy_instances(8)
    val_instances = toy_instances(4)

    function run(; kwargs...)
        Random.seed!(7)
        model = DiffusionModel(AGNN(cfg), BernoulliDiffusion(LinearScheduler(100, 3)))
        optim = Flux.setup(Flux.Adam(1e-3), model.gnn)
        loader = Flux.DataLoader(instances; batchsize=4, collate=true)
        val_loader = Flux.DataLoader(val_instances; batchsize=4, collate=true)
        return DiFoG.train_diffusion_model!(model, optim, loader; epochs=3, val_loader, kwargs...)
    end

    plain = run()
    tracker = RecordingTracker()
    tracked = run(; solver, val_graphs=first.(val_instances), val_batch_size=4, tracker)

    # measuring the solution quality changes neither the losses nor the random stream of the training
    @test tracked.batch_loss_history == plain.batch_loss_history
    @test tracked.val_loss_history == plain.val_loss_history

    # the first four return values are the ones of the previous positional interface
    losses, val_losses, batch_losses, val_batches = tracked
    @test length(losses) == 3 && length(val_losses) == 3 && length(batch_losses) == 6 && val_batches == [2, 4, 6]

    # per-epoch history with the solution quality, and without it when no solver is given
    @test length(tracked.epoch_history) == 3
    @test keys(tracked.epoch_history[1]) ==
          (:epoch, :batch, :train_loss, :val_loss, :val_gap, :val_objective, :val_heatmap_mae, :val_decisiveness)
    @test keys(plain.epoch_history[1]) == (:epoch, :batch, :train_loss, :val_loss)

    # the tracker receives the training loss of every batch and the validation metrics of every epoch
    steps = first.(tracker.logs)
    @test issorted(steps) && steps[end] == 6
    @test count(l -> haskey(l[2], "train/loss"), tracker.logs) == 6
    epoch_logs = [l[2] for l in tracker.logs if haskey(l[2], "val/gap")]
    @test length(epoch_logs) == 3
    @test all(haskey(epoch_logs[1], k) for k in ("epoch", "train/epoch_loss", "val/loss", "val/objective",
                                                  "val/heatmap_mae", "val/decisiveness"))
    @test epoch_logs[end]["val/gap"] == tracked.epoch_history[end].val_gap

    # the metric history is written as CSV, one row per epoch
    mktempdir() do dir
        path = save_metric_history(joinpath(dir, "metrics.csv"), tracked.epoch_history)
        lines = readlines(path)
        @test lines[1] == "epoch,batch,train_loss,val_loss,val_gap,val_objective,val_heatmap_mae,val_decisiveness"
        @test length(lines) == 4 && startswith(lines[2], "1,2,")
        loss_path = save_loss_history(tracked.batch_loss_history, tracked.val_loss_history,
                                      tracked.val_loss_batches, joinpath(dir, "loss.txt"), 4)
        @test isfile(loss_path) && readlines(loss_path)[3] == "batch,loss"
    end
end

@testitem "Test tracker backend is optional" begin
    # the W&B backend is a package extension: without Wandb loaded, only the function stub exists
    @test NoTracker() isa AbstractTracker
    @test log_metrics!(NoTracker(), Dict("train/loss" => 1.0); step=1) === nothing
    @test finish!(NoTracker()) === nothing
    @test isnothing(Base.get_extension(DiFoG, :DiFoGWandbExt)) == isempty(methods(WandbTracker))
end
