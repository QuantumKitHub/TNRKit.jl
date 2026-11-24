using Test
using TNRKit
using TensorKit

@testset "LoopTNR - 2D XY model" begin
    @info "Central charge of KT phase with U(1) symmetry"
    T_KT = classical_XY(U1Irrep, XY_βc + 0.1, 8)
    scheme = LoopTNR(T_KT)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = CFTData(scheme)
    central_charge = cft.central_charge
    @test central_charge ≈ 1.0 atol = 1.0e-2
    @info "Obtained central charge:\n$central_charge."

    @info "Central charge of symmetric phase with U(1) symmetry"
    T_sym = classical_XY(U1Irrep, XY_βc - 0.1, 8)
    scheme = LoopTNR(T_sym)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = CFTData(scheme)
    central_charge = cft.central_charge
    @test central_charge ≈ 0.0 atol = 1.0e-10
    @info "Obtained central charge:\n$central_charge."

    @info "Central charge of KT phase with O(2) symmetry"
    T_KT = classical_XY(CU1Irrep, XY_βc + 0.1, 8)
    scheme = LoopTNR(T_KT)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = CFTData(scheme)
    central_charge = cft.central_charge
    @test central_charge ≈ 1.0 atol = 1.0e-2
    @info "Obtained central charge:\n$central_charge."

    @info "Central charge of symmetric phase with O(2) symmetry"
    T_sym = classical_XY(CU1Irrep, XY_βc - 0.1, 8)
    scheme = LoopTNR(T_sym)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = CFTData(scheme)
    central_charge = cft.central_charge
    @test central_charge ≈ 0.0 atol = 1.0e-10
    @info "Obtained central charge:\n$central_charge."
end
