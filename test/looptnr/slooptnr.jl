using Test
using TNRKit
using TensorKit

# SLoopTNR
@testset "SLoopTNR - Ising Model" begin
    @info "SLoopTNR ising free energy"
    T_inv = classical_ising_inv()
    scheme = SLoopTNR(T_inv)

    data = run!(scheme, truncrank(4), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-5
end
