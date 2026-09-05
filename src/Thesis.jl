module Thesis

using CUDA
using Flux
using GraphNeuralNetworks
using Statistics
using DotEnv
using Logging
using MHLib

DotEnv.load!(joinpath(@__DIR__, ".env"))

include("./utils/logging_utils.jl")

include("./diffusion/diffusion.jl")
include("./diffusion/scheduler.jl")
include("./gnn/gnn.jl")

include("./gnn/gnn_config.jl") 
include("./utils/math_utils.jl")

include("diffusion_model.jl")

include("./utils/model_utils.jl")
include("./utils/decoding_utils.jl")

include("./diffusion/methods/diffusion/bernoulli_diffusion.jl")
include("./diffusion/methods/flow_matching/discrete_flow_matching.jl")

include("./utils/data/data_utils.jl")
include("./utils/data/runtime_dataset.jl")

include("./diffusion/scheduler/linear.jl")
include("./diffusion/scheduler/cosine.jl")
include("./diffusion/scheduler/atss.jl")

include("./gnn/models/AGNN.jl")
include("./graphs/er.jl")
include("./graphs/basic.jl")

include("./problems/misp.jl")
include("./problems/maxdeg.jl")

abstract type Config end

include("./problems/NodeSelectionProblems.jl")
include("./problems/maxcut.jl")

function __init__()
    DotEnv.load!(joinpath(@__DIR__, ".env"))
    set_log_level!() # for loglevel
    _init_gurobi_env!() # init gurobi all the way in the beginning, else sometimes there is strange behavior...
    return
end

end