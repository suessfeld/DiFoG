module DiFoG

using CUDA
using Flux
using GraphNeuralNetworks
using Statistics
using Logging
using MHLib
using Random
using Plots


include("./utils/logging_utils.jl")
include("./utils/gurobi_session.jl")

include("./diffusion/diffusion.jl")
include("./diffusion/scheduler.jl")
include("./gnn/gnn.jl")

include("./gnn/gnn_config.jl") 

include("diffusion_model.jl")

include("./utils/model_utils.jl")
include("./utils/decoding_utils.jl")
include("./utils/plotting.jl")

include("./diffusion/methods/diffusion/bernoulli_diffusion.jl")
include("./diffusion/methods/flow_matching/discrete_flow_matching.jl")

include("./utils/data/data_utils.jl")
include("./utils/data/runtime_dataset.jl")
include("./utils/data/seeding.jl")  

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
    set_log_level!() # for loglevel
    _init_gurobi_env!() # init gurobi all the way in the beginning, else sometimes there is strange behavior...
    return
end

end