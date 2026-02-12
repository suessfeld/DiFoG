
abstract type GNN end

function train(model::GNN)
    error("train() has not been implemented for type $(typeof(model))")
end