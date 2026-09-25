using TestItems
using DiFoG

@testmodule TestRectifiedFlow begin
    using Test
    using DiFoG
    using Flux
    using GraphNeuralNetworks

    export toy_graphs, toy_batch

    # A labelled 4-cycle and a labelled triangle (7 nodes in total).
    toy_graphs() = [
        GNNGraph(Int32[1, 2, 3, 4, 2, 3, 4, 1], Int32[2, 3, 4, 1, 1, 2, 3, 4]; num_nodes=4,
                 ndata = (; x = Float32[1 0 1 0])),
        GNNGraph(Int32[1, 2, 3, 2, 3, 1], Int32[2, 3, 1, 1, 2, 3]; num_nodes=3,
                 ndata = (; x = Float32[1 0 0])),
    ]
    toy_batch() = Flux.batch(toy_graphs())
end

@testitem "Test RectifiedFlow construction and schedule conversion" setup=[TestRectifiedFlow] begin
    rf = RectifiedFlow(LinearScheduler(1000, 4))
    @test eltype(typeof(rf.scheduler)) == Float32
    @test rf.prediction === :x1 && rf.σ_src === 1f0
    @test rf.scheduler.steps == Float32[0, 0.25, 0.5, 0.75, 1]
    @test RectifiedFlow(LinearScheduler(1000, 4); prediction=:velocity).prediction === :velocity
    @test_throws ArgumentError RectifiedFlow(LinearScheduler(1000, 4); prediction=:eps)
    @test_throws ArgumentError RectifiedFlow(LinearScheduler(1000, 4); σ_src=0)
end

@testitem "Test RectifiedFlow corrupt_batch lies on the straight path" setup=[TestRectifiedFlow] begin
    rf = RectifiedFlow(LinearScheduler(100, 10))
    g = toy_batch()
    y = g.ndata.x

    # independent coupling: x_0 is sampled and returned in aux
    x_t, t, y_out, aux = DiFoG.corrupt_batch(rf, y, g)
    @test !aux.coupled && y_out === y
    τ = reshape(Float32.(t[g.graph_indicator]) ./ 100f0, 1, :)
    @test x_t ≈ τ .* (2f0 .* y .- 1f0) .+ (1f0 .- τ) .* aux.x_0

    # stored Reflow coupling: x_0 is taken from ndata.x0
    x0 = reshape(Float32.(1:7), 1, :)
    gc = GNNGraph(g; ndata = (; x = y, x0))
    x_t, t, _, aux = DiFoG.corrupt_batch(rf, y, gc)
    @test aux.coupled && aux.x_0 == x0
    τ = reshape(Float32.(t[gc.graph_indicator]) ./ 100f0, 1, :)
    @test x_t ≈ τ .* (2f0 .* y .- 1f0) .+ (1f0 .- τ) .* x0
end

@testitem "Test RectifiedFlow Euler steps (both parameterizations)" setup=[TestRectifiedFlow] begin
    g = toy_batch()
    x = Float32[0.3 -1.2 0.5 2.0 -0.1 0.0 1.0]

    # :x1 — the endpoint logit fixes x̂₁; a step to τ = 1 lands exactly on x̂₁
    rf = RectifiedFlow(LinearScheduler(100, 4))
    logits = Float32[3 -3 0 1 -1 2 -2]
    model = (x_, t_, g_) -> logits
    x̂1 = 2f0 .* Flux.sigmoid.(logits) .- 1f0
    x_next, x̂1_out, v = DiFoG.p(rf, model, x, 0.5f0, 0.75f0, g)
    @test x̂1_out ≈ x̂1
    @test v ≈ (x̂1 .- x) ./ 0.5f0
    @test x_next ≈ x .+ 0.25f0 .* v
    @test DiFoG.p(rf, model, x, 0.5f0, 1f0, g)[1] ≈ x̂1

    # :velocity — a constant (perfectly straight) oracle velocity is integrated exactly by ANY grid
    rfv = RectifiedFlow(LinearScheduler(100, 4); prediction=:velocity)
    x_0 = randn(Float32, 1, 7)
    x_1 = Float32[1 -1 1 -1 1 -1 -1]
    oracle = (x_, t_, g_) -> x_1 .- x_0
    @test DiFoG.integrate(rfv, oracle, x_0, g, Float32[0, 1]) ≈ x_1
    @test DiFoG.integrate(rfv, oracle, x_0, g, rfv.scheduler.steps) ≈ x_1
end

@testitem "Test RectifiedFlow sample_heatmap returns a heatmap in [0, 1]" setup=[TestRectifiedFlow] begin
    g = toy_batch()
    rf = RectifiedFlow(LinearScheduler(100, 3))
    logits = Float32[3 -3 0 1 -1 2 -2]
    h = DiFoG.sample_heatmap(rf, (x_, t_, g_) -> logits, (1, 7), g)
    @test size(h) == (1, 7)
    @test h ≈ Flux.sigmoid.(logits)   # the last :x1 step lands on x̂₁ = 2σ − 1

    rfv = RectifiedFlow(LinearScheduler(100, 3); prediction=:velocity)
    h = DiFoG.sample_heatmap(rfv, (x_, t_, g_) -> fill(100f0, 1, 7), (1, 7), g)
    @test all(h .== 1f0)              # clamped
end

@testitem "Test RectifiedFlow straightness is zero for an exactly straight flow" setup=[TestRectifiedFlow] begin
    g = toy_batch()
    rfv = RectifiedFlow(LinearScheduler(100, 3); prediction=:velocity)
    s = straightness(rfv, (x_, t_, g_) -> fill(0.7f0, 1, 7), g, (1, 7); ode_steps=20)
    @test s.straightness ≈ 0f0 atol=1e-10
    @test s.one_step_error ≈ 0f0 atol=1e-10

    # a curved flow (velocity depends on time) has positive straightness
    s = straightness(rfv, (x_, t_, g_) -> fill(Float32(t_[1]) / 100f0, 1, 7), g, (1, 7); ode_steps=20)
    @test s.straightness > 0f0 && s.one_step_error > 0f0
end

@testitem "Test RectifiedFlow losses, gradients and Reflow round trip with AGNN" setup=[TestRectifiedFlow] begin
    using Random
    Random.seed!(1)
    g = toy_batch()
    cfg = GNNConfig(num_layers=2, in_dim=1, time_dim=8, hidden_dim=16, output_dim=1)

    for prediction in (:x1, :velocity), symmetric in (false, true)
        rf = RectifiedFlow(LinearScheduler(100, 5); prediction)
        model = DiffusionModel(AGNN(cfg), rf)
        x_t, t, y, aux = DiFoG.corrupt_batch(rf, g.ndata.x, g)
        parts, grads = DiFoG.compute_loss_and_gradients(model.gnn, rf, x_t, t, g, y, aux; symmetric)
        @test isfinite(parts.loss) && parts.loss > 0
        @test !isnothing(grads[1])
        # flip-invariant loss can only be lower
        @test DiFoG.compute_loss(model.gnn, rf, x_t, t, g, y, aux; symmetric=true).loss ≤
              DiFoG.compute_loss(model.gnn, rf, x_t, t, g, y, aux; symmetric=false).loss + 1f-6
    end

    # Reflow: coupling generation keeps graphs/names, stores x0 and a soft Z₁; training runs on it
    rf = RectifiedFlow(LinearScheduler(100, 5))
    model = DiffusionModel(AGNN(cfg), rf)
    instances = [(gr, "inst_$i") for (i, gr) in enumerate(vcat(toy_graphs(), toy_graphs(), toy_graphs()))]
    loader = Flux.DataLoader(instances; batchsize=4, collate=true)
    coupling = generate_reflow_coupling(model, loader; ode_steps=8)
    @test last.(coupling) == last.(instances)
    @test [gc.num_nodes for (gc, _) in coupling] == [gr.num_nodes for (gr, _) in instances]
    gc, _ = coupling[1]
    @test :x0 in keys(gc.ndata) && size(gc.ndata.x0) == (1, 4)
    @test all(0f0 .< gc.ndata.x .< 1f0)   # soft labels from the :x1 endpoint

    # the stored Z₁ is exactly the ODE endpoint of the stored x₀ (same batch composition as during
    # generation: BatchNorm(track_stats=false) normalizes with the statistics of the current batch)
    gb = Flux.batch(first.(coupling)[1:4])
    z1 = DiFoG.integrate(rf, model, gb.ndata.x0, gb, DiFoG._uniform_grid(8))
    @test gb.ndata.x ≈ (z1 .+ 1f0) ./ 2f0

    # coupled batches disable the flip-invariant loss
    gb = Flux.batch(first.(coupling))
    x_t, t, y, aux = DiFoG.corrupt_batch(rf, gb.ndata.x, gb)
    @test DiFoG.compute_loss(model.gnn, rf, x_t, t, gb, y, aux; symmetric=true).loss ≈
          DiFoG.compute_loss(model.gnn, rf, x_t, t, gb, y, aux; symmetric=false).loss

    optim = Flux.setup(Flux.Adam(1e-3), model.gnn)
    losses, _, batch_losses, _ = reflow!(model, optim, loader; ode_steps=4, epochs=2, batch_size=2)
    @test length(losses) == 2 && all(isfinite, batch_losses)
end
