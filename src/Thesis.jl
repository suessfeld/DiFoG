module Thesis

using Flux
using GraphNeuralNetworks
using Statistics
using DotEnv

DotEnv.load!()

# 1. Base Types (Must be loaded first)
include("./diffusion/diffusion.jl") # Defines abstract type Diffusion
include("./gnn/gnn.jl")             # Defines abstract type GNN

# 2. Configs & Utils
include("./gnn/gnn_config.jl") 
include("./utils/math_utils.jl")

# 3. Core Wrappers (Depends on Base Types)
include("diffusion_model.jl")       # Depends on Diffusion and GNN

# 4. Implementations (Methods and Models)
include("./diffusion/methods/bernoulli_diffusion.jl")
include("./diffusion/methods/gaussian_diffusion.jl")
include("./gnn/models/anisotropic_gnn.jl")

end