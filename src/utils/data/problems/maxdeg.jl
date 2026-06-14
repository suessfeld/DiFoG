using Graphs

export MaxDegSolver

struct MaxDegSolver <: ProblemSolver end

function (::MaxDegSolver)(g::Graphs.SimpleGraph)
    v = nv(g)
    
    degs = degree(g)
    max_deg = maximum(degs)
    max_deg_nodes = findall(x -> x == max_deg, degs)

    x_0 = zeros(Float32, 1, v)
    for idx in max_deg_nodes
        x_0[1, idx] = 1.0f0
    end
    
    return x_0
end