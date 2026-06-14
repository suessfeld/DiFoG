module Thesis

using CUDA
using Flux
using GraphNeuralNetworks
using Statistics
using DotEnv

DotEnv.load!()

include("./diffusion/diffusion.jl")
include("./gnn/gnn.jl")

include("./gnn/gnn_config.jl") 
include("./utils/math_utils.jl")

include("diffusion_model.jl")

include("./utils/model_utils.jl")
include("./utils/decoding_utils.jl")

include("./diffusion/methods/bernoulli_diffusion.jl")
include("./diffusion/methods/gaussian_diffusion.jl")
include("./gnn/models/anisotropic_gnn_basic.jl")
include("./gnn/models/difusco_gnn.jl")

include("./utils/data/data_utils.jl")
include("./utils/data/runtime_dataset.jl")

include("./utils/data/graphs/er.jl")
include("./utils/data/graphs/basic.jl")

include("./utils/data/problems/misp.jl")
include("./utils/data/problems/maxdeg.jl")

end