module Thesis

using Flux
using GraphNeuralNetworks
using Statistics

include("./gnn/gnn.jl")
include("./gnn/gnn_config.jl")

include("./gnn/anisotropic/anisotropic_gnn.jl")

include("./tests/misp/misp_solver.jl")
include("./tests/unit/test_anisotropic_gnn.jl")
include("./tests/testrunner.jl")

include("./utils/data_utils.jl")
include("./utils/generate_testcases.jl")
include("./utils/math_utils.jl")
end 