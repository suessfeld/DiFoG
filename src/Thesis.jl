module Thesis

using CUDA
using Flux
using GraphNeuralNetworks
using Statistics
using DotEnv

DotEnv.load!(joinpath(@__DIR__, ".env"))

include("./diffusion/diffusion.jl")
include("./diffusion/scheduler.jl")
include("./gnn/gnn.jl")

include("./gnn/gnn_config.jl") 
include("./utils/math_utils.jl")

include("diffusion_model.jl")

include("./utils/model_utils.jl")
include("./utils/decoding_utils.jl")

include("./diffusion/methods/diffusion/bernoulli_diffusion.jl")
include("./diffusion/methods/diffusion/gaussian_diffusion.jl")
include("./diffusion/methods/flow_matching/discrete_flow_matching.jl")

include("./diffusion/scheduler/linear.jl")
include("./diffusion/scheduler/cosine.jl")
include("./diffusion/scheduler/atss.jl")

include("./gnn/models/anisotropic_gnn_basic.jl")
include("./gnn/models/difusco_gnn.jl")

include("./utils/data/data_utils.jl")
include("./utils/data/runtime_dataset.jl")

include("./utils/data/graphs/er.jl")
include("./utils/data/graphs/basic.jl")

include("./utils/data/problems/misp.jl")
include("./utils/data/problems/maxdeg.jl")

function __init__()
    DotEnv.load!(joinpath(@__DIR__, ".env"))
    _init_gurobi_env!()
    return
end

end