using TestItems
using Thesis

@testmodule TestMathUtils begin
    using Test
    using Thesis
end

@testitem "Test adjacency_list_to_matrix basic conversion" setup=[TestMathUtils] begin
    # NOTE: adjacency_list_to_matrix is not exported from Thesis, so it must
    # be accessed via the qualified module path.
    adj_list = [[2, 3], [1], [1]]

    edge_matrix = Thesis.adjacency_list_to_matrix(adj_list)

    # adjacency_list_to_matrix emits one directed (src,dst) pair per
    # adjacency-list entry -- it does NOT deduplicate or treat the list as
    # undirected. Node 1 lists 2 neighbors, nodes 2 and 3 each list 1, for
    # 4 total directed edges: (1,2), (1,3), (2,1), (3,1).
    @test size(edge_matrix) == (2, 4)
    # Row 1 = sources, row 2 = destinations (pre-transpose columns become rows)
    @test edge_matrix[1, :] == [1, 1, 2, 3]
    @test edge_matrix[2, :] == [2, 3, 1, 1]
end

@testitem "Test adjacency_list_to_matrix empty adjacency list" setup=[TestMathUtils] begin
    adj_list = Vector{Vector{Int}}([[], [], []])

    edge_matrix = Thesis.adjacency_list_to_matrix(adj_list)

    @test size(edge_matrix, 2) == 0
end

@testitem "Test adjacency_list_to_matrix preserves edge count" setup=[TestMathUtils] begin
    adj_list = [[2], [3], [1, 2]]
    total_edges = sum(length, adj_list)

    edge_matrix = Thesis.adjacency_list_to_matrix(adj_list)

    @test size(edge_matrix, 2) == total_edges
end