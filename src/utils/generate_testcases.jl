# TODO: Move this file

nodes = 10
edges = 20
graph_label = 1
output_file = "./src/tests/cases/graph_$(nodes)_$(edges)_$(graph_label).txt"

# Run the generator
generate_graph(output_file, num_nodes=nodes, num_edges=edges, label=graph_label)