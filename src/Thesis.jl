module Thesis

include("./gnn/gnn.jl")
include("./gnn/layer.jl")
include("./gnn/gnn_config.jl")

include("./gnn/anisotropic/anisotropic_gnn_layer.jl")
include("./gnn/anisotropic/anisotropic_gnn.jl")

include("./tests/misp/misp_solver.jl")
include("./tests/unit/test_anisotropic_gnn.jl")
include("./tests/testrunner.jl")

include("./utils/data_utils.jl")
include("./utils/diffusion_utils.jl")
include("./utils/generate_testcases.jl")
include("./utils/math_utils.jl")
end 