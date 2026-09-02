module Thesis  #GR final sollten wir hier ein hübsches Acronym für das Framework haben

using CUDA
using Flux
using GraphNeuralNetworks
using Statistics
using DotEnv
using Logging
using MHLib

DotEnv.load!(joinpath(@__DIR__, ".env"))  #GR TODO: Besser anders lösen, z.B. in Ihrem .profile, hier keine User-abhängigen Pfade

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
include("./diffusion/methods/diffusion/gaussian_diffusion.jl")  #GR TODO Does not exist!!
include("./diffusion/methods/flow_matching/discrete_flow_matching.jl")

include("./utils/data/data_utils.jl")
include("./utils/data/runtime_dataset.jl")

include("./diffusion/scheduler/linear.jl")
include("./diffusion/scheduler/cosine.jl")
include("./diffusion/scheduler/atss.jl")

include("./gnn/models/AGNN.jl")

#GR TODO Bitte nur die Graphgeneratoren in NodeSelectionProblems.jl verwenden, keine "doppelte" Funktionalität.

#GR TODO Es ist unintuitiv, dass problems (und graphs) in einem Unterverzeichnis data liegen, ich würde NodeSelection Problems.jl entweder einfach in src legen oder in src/problems.
include("./utils/data/graphs/er.jl")
include("./utils/data/graphs/basic.jl")

include("./utils/data/problems/misp.jl")
include("./utils/data/problems/maxdeg.jl")

abstract type Config end

include("./utils/data/problems/NodeSelectionProblems.jl")
include("./utils/data/problems/maxcut.jl")

function __init__()
    DotEnv.load!(joinpath(@__DIR__, ".env"))
    set_log_level!() # for loglevel
    _init_gurobi_env!() # init gurobi all the way in the beginning, else sometimes there is strange behavior...
    return
end

end