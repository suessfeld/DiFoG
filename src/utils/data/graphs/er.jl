using Graphs

export ERGraphGenerator

struct ERGraphGenerator <: GraphGenerator
    min_nodes::Int
    max_nodes::Int
    p::Float64
end

# Keyword constructor with default values
function ERGraphGenerator(; min_nodes::Int=50, max_nodes::Int=100, p::Float64=0.15)
    return ERGraphGenerator(min_nodes, max_nodes, p)
end

function (gen::ERGraphGenerator)()
    v = rand(gen.min_nodes:gen.max_nodes)
    
    g = erdos_renyi(v, gen.p)
    if nv(g) == 0
        g = SimpleGraph(v)
    end
    
    return g
end