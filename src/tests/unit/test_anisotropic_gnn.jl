using TestItems

@testitem "Anisotropic GNN Forward Pass" setup=[utils] begin
    # First do it manually
    U = Float32[1 2 1; 
        0 1 1; 
        2 1 1]

    V = Float32[0 0 1; 
        1 2 1; 
        0 1 1]

    A = Float32[1 0.5 1; 
        2 2 1; 
        1 1 1]

    B = Float32[1 0 0; 
        0 1 0; 
        0 0 1]

    h_1 = [1, 0, 0]
    h_2 = [0, 1, 0]
    h_3 = [0, 0, 1]
    h_4 = [1, 1, 1]

    h_1_new = relu(U*h_1 + (han(sig(A*h_1 + B*h_2), V*h_2)))
    h_2_new = relu(U*h_2 + (han(sig(A*h_2 + B*h_1), V*h_1) + han(sig(A*h_2 + B*h_3), V*h_3) + han(sig(A*h_2 + B*h_4), V*h_4)))
    h_3_new = relu(U*h_3 + (han(sig(A*h_3 + B*h_2), V*h_2) + han(sig(A*h_3 + B*h_4), V*h_4)))
    h_4_new = relu(U*h_4 + (han(sig(A*h_4 + B*h_2), V*h_2) + han(sig(A*h_4 + B*h_3), V*h_3)))

    manual_out = [h_1_new h_2_new h_3_new h_4_new]
    print(manual_out)
    
    # Now with implementation
    custom_layer = AnisotropicGNNLayer(U, V, A, B)

    # Just 1 layer with 3x3 matrices
    config = GNNConfig(1,3,3,3)
    model = AnisotropicGNN(config, [custom_layer])
    x = Float32.(hcat(h_1, h_2, h_3, h_4))
    
    edge_index = [
        2 1 3 4 2 4 2 3; # src (j)
        1 2 2 2 3 3 4 4  # dst (i)
    ]

    # Forward pass
    gnn_out = model(x, edge_index)
    @test isapprox(gnn_out, manual_out, atol=1e-6)
end

@testsnippet utils begin

    using Thesis: AnisotropicGNN, GNNConfig, AnisotropicGNNLayer

    function sig(x)
        return 1 ./ (1 .+ exp.(-x))
    end

    function relu(x)
        return max.(0, x)
    end

    function han(x, y)
        return x .* y
    end
end