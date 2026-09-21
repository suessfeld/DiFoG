using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CUDA
using cuDNN
using Flux
using Thesis

include(joinpath(pkgdir(Thesis), "test", "test_utils.jl"))


result_root = "/home/e12023149/thesis/results"
current_file = basename(@__FILE__)
run_id = replace(current_file, ".jl" => "")

generator = ERGraphGenerator(min_nodes=300, max_nodes=400, p=0.15)
solver = MaxCutSolver(time_limit=60.0)

seed = 30040001
dataset_path = create_dataset(seed, joinpath(result_root, "seeds"), generator, solver, 10000)
@info "Seeded dataset written to $dataset_path"
