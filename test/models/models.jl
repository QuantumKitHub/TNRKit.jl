println("--------------------")
println(" Testing all models ")
println("--------------------")

model_temp_answer_string_2d = [
    (classical_ising(Trivial), ising_βc, f_onsager, "2D Ising model with no symmetry"),
    (classical_ising(), ising_βc, f_onsager, "2D Ising model with ℤ₂ symmetry"),
    (gross_neveu_start(0, 0, 0), 1.0, -1.4515448845652446, "Gross-Neveu model"),
    (classical_clock(Trivial, 3, 2.0 * log(√3 + 1) / 3), 2.0 * log(√3 + 1) / 3, -4.17924244901635, "3-state clock model with no symmetry"), # This is an approximation!
    (classical_clock(Z3Irrep, 3, 2.0 * log(√3 + 1) / 3), 2.0 * log(√3 + 1) / 3, -4.17924244901635, "3-state clock model with ℤ₃ symmetry"), # This is an approximation!
    (classical_clock(D3Irrep, 3, 2.0 * log(√3 + 1) / 3), 2.0 * log(√3 + 1) / 3, -4.17924244901635, "3-state clock model with D₃ symmetry"), # This is an approximation!
    (classical_clock(Trivial, 4, log(√2 + 1)), log(√2 + 1), 2 * f_onsager, "4-state clock model with no symmetry"), # It can be proved that 4-state clock model is equivalent to two layers of Ising model.
    (classical_clock(Z4Irrep, 4, log(√2 + 1)), log(√2 + 1), 2 * f_onsager, "4-state clock model with ℤ₄ symmetry"),
    (classical_clock(D4Irrep, 4, log(√2 + 1)), log(√2 + 1), 2 * f_onsager, "4-state clock model with D₄ symmetry"),
    (classical_potts(Trivial, 3), potts_βc(3), -4.119552029995684, "Potts model with no symmetry"), # This is an approximation!
    (classical_potts(3), potts_βc(3), -4.119552029995684, "Potts model with ℤ₃ symmetry"), # This is an approximation!
    (sixvertex(Trivial), 1.0, 3 / 2 * log(3 / 4), "Six-vertex model with no symmetry"),
    (sixvertex(U1Irrep), 1.0, 3 / 2 * log(3 / 4), "Six-vertex model with U(1) symmetry"),
    (sixvertex(), 1.0, 3 / 2 * log(3 / 4), "Six-vertex model with CU(1) symmetry"),
    # (classical_XY(U1Irrep, 0.89351, 6), 0.89351, -1.0251, "Classical XY model with U(1) symmetry"), # This is an approximation!
    # (classical_XY(CU1Irrep, 0.89351, 6), 0.89351, -1.0251, "Classical XY model with CU(1) symmetry"), # This is an approximation!
    (phi4_real(Trivial, 10, -1.0, 1.0), -1.0, 0.4241912271276211, "Real φ⁴ model with no symmetry"), # This is an approximation!
    (phi4_real(10, -1.0, 1.0), -1.0, 0.4232381701937374, "Real φ⁴ model with ℤ₂ symmetry"), # This is an approximation!
    (phi4_complex(Trivial, 6, -1.0, 1.0), -1.0, 0.7583605364656325, "Complex φ⁴ model with no symmetry"), # This is an approximation!
    (phi4_complex(6, -1.0, 1.0), -1.0, 0.7673189874157453, "Complex φ⁴ model with U(1) symmetry"), # This is an approximation!
    (phi4_complex(Z2Irrep ⊠ Z2Irrep, 6, -1.0, 1.0), -1.0, 0.7665677554973079, "Complex φ⁴ model with ℤ₂ × ℤ₂ symmetry"), # This is an approximation!
]

model_temp_answer_string_3d = [
    (classical_ising_3D(Trivial), ising_βc_3D, -3.508, "3D Ising model with no symmetry"), # This is an approximation!
    (classical_ising_3D(), ising_βc_3D, -3.508, "3D Ising model with ℤ₂ symmetry"), # This is an approximation!
]

for (model, temp, answer, description) in model_temp_answer_string_2d
    @testset "$(description)" begin
        scheme = TRG(model)
        data = run!(scheme, truncrank(16), maxiter(25))
        @test free_energy(data, temp) ≈ answer rtol = 1.0e-3
    end
end

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
    @test central_charge ≈ 0.0 atol = 1.0e-13
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
    @test central_charge ≈ 0.0 atol = 1.0e-13
    @info "Obtained central charge:\n$central_charge."
end

for (model, temp, answer, description) in model_temp_answer_string_3d
    @testset "$(description)" begin
        scheme = HOTRG_3D(model)
        data = run!(scheme, truncrank(8), maxiter(25))
        @test free_energy(data, temp; scalefactor = 8.0) ≈ answer rtol = 1.0e-3
    end
end


# Test for impurity tensors
# Real phi^4
@testset "Real φ⁴ model - Impure" begin
    # Disordered phase
    λ = 1.0
    μ0 = 0.0

    Tpure = phi4_real(Trivial, 10, μ0, λ)
    T_imp1 = phi4_real_imp1(Trivial, 10, μ0, λ)
    T_imp2 = phi4_real_imp2(Trivial, 10, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp1, T_imp1, T_imp2)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][4] / data[end][1]
    @test order_para ≈ 0.0 atol = 1.0e-3

    # Ordered phase
    μ0 = -2.0

    Tpure = phi4_real(Trivial, 10, μ0, λ)
    T_imp1 = phi4_real_imp1(Trivial, 10, μ0, λ)
    T_imp2 = phi4_real_imp2(Trivial, 10, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp1, T_imp1, T_imp2)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][4] / data[end][1]
    @test order_para ≈ 1.5317112652447245 rtol = 1.0e-3
end

# Complex phi^4
@testset "Complex φ⁴ model - Impure" begin
    # Disordered phase
    λ = 1.0
    μ0 = 0.0

    Tpure = phi4_complex(Trivial, 6, μ0, λ; T = ComplexF64)
    T_imp11 = phi4_complex_impϕ(6, μ0, λ)
    T_imp12 = phi4_complex_impϕdag(6, μ0, λ)
    T_imp2 = phi4_complex_impϕ2(6, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp11, T_imp12, T_imp2)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][4] / data[end][1]
    @test order_para ≈ 0.0 atol = 1.0e-3
end
