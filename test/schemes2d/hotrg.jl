using Test
using TNRKit
using TensorKit

T = classical_ising()

# HOTRG
@testset "HOTRG - Ising Model" begin
    @info "HOTRG ising free energy"
    scheme = HOTRG(T)
    data = run!(scheme, truncrank(16), maxiter(25))

    @test free_energy(data, ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 6.0e-7

    @info "HOTRG ising CFT data"
    scheme = HOTRG(T)
    run!(scheme, truncrank(16), maxiter(4))

    cft = sort(CFTData(scheme; shape = [1, 1, 0]).scaling_dimensions[2:end]; by = abs) .|> real

    @test cft[1] ≈ ising_cft_exact[1] rtol = 6.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2

    @info "HOTRG ising ground state degeneracy"
    T1 = classical_ising(ising_βc - 0.01)
    scheme = HOTRG(T1)
    run!(scheme, truncrank(12), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 1 rtol = 1.0e-2
    @test X1 ≈ 1.0 rtol = 1.0e-2
    @test X2 ≈ 1.0 rtol = 1.0e-2

    T2 = classical_ising(ising_βc + 0.01)
    scheme = HOTRG(T2)
    run!(scheme, truncrank(12), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 2 rtol = 1.0e-2
    @test X1 ≈ 2.0 rtol = 1.0e-2
    @test X2 ≈ 2.0 rtol = 1.0e-2
end

# ATRG
@testset "ATRG - Ising Model" begin
    @info "ATRG ising free energy"
    scheme = ATRG(T)
    data = run!(scheme, truncrank(24), maxiter(25))

    @test free_energy(data, ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 3.0e-6

    @info "ATRG ising CFT data"
    scheme = ATRG(T)
    run!(scheme, truncrank(24), maxiter(3))

    cft = sort(CFTData(scheme; shape = [1, 1, 0]).scaling_dimensions[2:end]; by = abs) .|> real

    @test cft[1] ≈ ising_cft_exact[1] rtol = 1.0e-2
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2

    @info "ATRG ising ground state degeneracy"
    T1 = classical_ising(ising_βc - 0.01)
    scheme = ATRG(T1)
    run!(scheme, truncrank(16), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 1 rtol = 1.0e-2
    @test X1 ≈ 1.0 rtol = 1.0e-2
    @test X2 ≈ 1.0 rtol = 1.0e-2

    T2 = classical_ising(ising_βc + 0.01)
    scheme = ATRG(T2)
    run!(scheme, truncrank(16), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 2 rtol = 1.0e-2
    @test X1 ≈ 2.0 rtol = 1.0e-2
    @test X2 ≈ 2.0 rtol = 1.0e-2
end
