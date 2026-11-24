using Test
using TNRKit
using TensorKit
using TensorKitSectors

# Pre-computed anisotropic Ising critical values for testing (Jx=1, Jy=0.5)
const ising_aniso_βc_1_05 = 0.6093778634360062
const f_onsager_aniso_1_05 = -1.5741477440817498

model_temp_answer_string_2d = [
    (classical_ising(Trivial), ising_βc, f_onsager, "2D Ising model with no symmetry"),
    (classical_ising(), ising_βc, f_onsager, "2D Ising model with ℤ₂ symmetry"),
    (classical_ising(ising_aniso_βc_1_05; Jx = 1.0, Jy = 0.5), ising_aniso_βc_1_05, f_onsager_aniso_1_05, "2D anisotropic Ising model Jx=1,Jy=0.5 with ℤ₂ symmetry"),
    (classical_ising(Trivial, ising_aniso_βc_1_05; Jx = 1.0, Jy = 0.5), ising_aniso_βc_1_05, f_onsager_aniso_1_05, "2D anisotropic Ising model Jx=1,Jy=0.5 with no symmetry"),
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
]

for (model, temp, answer, description) in model_temp_answer_string_2d
    @testset "$(description)" begin
        scheme = TRG(model)
        data = run!(scheme, truncrank(16), maxiter(25))
        @test free_energy(data, temp) ≈ answer rtol = 1.0e-3
    end
end

# Tests for anisotropic Ising helper functions
@testset "Anisotropic Ising helpers" begin
    # Critical condition: sinh(2βc Jx) · sinh(2βc Jy) = 1, and βc scaling
    for (Jx, Jy) in [(1.0, 1.0), (1.0, 0.5), (1.0, 0.3), (2.0, 2.0)]
        βc = ising_anisotropic_βc(Jx, Jy)
        @test sinh(2 * Float64(βc) * Jx) * sinh(2 * Float64(βc) * Jy) ≈ 1.0 rtol = 1.0e-14
        if Jx == Jy
            @test Float64(βc) ≈ Float64(ising_βc) / Jx  # βc scales as 1/J
        end
    end

    # Free energy scaling: f(β, J, J) = J · f_iso(β·J)
    for (β, J) in [(Float64(ising_βc), 1.0), (0.5, 2.0), (0.3, 1.5)]
        βc = ising_anisotropic_βc(J, J)
        f_J = f_onsager_anisotropic(β, J, J)
        f_Jc = f_onsager_anisotropic(βc, J, J)
        f_1 = f_onsager_anisotropic(β * J, 1.0, 1.0)
        @test f_J ≈ J * f_1 rtol = 1.0e-6
        @test f_Jc ≈ J * Float64(f_onsager) rtol = 1.0e-6
    end

    # Jy→0 limit: f should approach the 1D Ising result
    β = 0.5
    f_1d = -(log(2.0) + log(cosh(β))) / β
    @test f_onsager_anisotropic(β, 1.0, 1.0e-10) ≈ f_1d rtol = 1.0e-6
end

@testset "Six-vertex model - Crosscheck" begin
    a, b, c = 1.0, 2.0, 3.0
    Ttriv = convert(Array, sixvertex(Trivial; a, b, c))
    Tu1 = convert(Array, sixvertex(U1Irrep; a, b, c))
    Tcu1 = convert(Array, sixvertex(CU1Irrep; a, b, c))
    @test iszero(norm(Ttriv - Tu1))
    @test iszero(norm(Ttriv - Tcu1))
    @test iszero(norm(Tu1 - Tcu1))
end
