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

generator = ERGraphGenerator(min_nodes=700, max_nodes=800, p=0.15)
solver = MISPSolver(time_limit=1.0)

seed = 70080003
dataset_path = create_dataset(seed, joinpath(result_root, "seeds"), generator, solver, 100)
@info "Seeded dataset written to $dataset_path"
