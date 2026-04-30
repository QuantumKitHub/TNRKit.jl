f_bench = -1.4515448845652446
T = gross_neveu_start(0, 0, 0)

# === TRG ===
@testset "TRG - Gross-Neveu Model" begin
    scheme = TRG(T)
    data = run!(scheme, truncrank(16), maxiter(25))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-3
end

# === BTRG ===
@testset "BTRG - Gross-Neveu Model" begin
    scheme = BTRG(T)
    data = run!(scheme, truncrank(16), maxiter(25))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-4
end

# === HOTRG ===
@testset "HOTRG - Gross-Neveu Model" begin
    scheme = HOTRG(T)
    data = run!(scheme, truncrank(16), maxiter(25))
    @test free_energy(data, 1.0; scalefactor = 4.0) ≈ f_bench rtol = 1.0e-3
end

# === ATRG ===
@testset "ATRG - Gross-Neveu Model" begin
    scheme = ATRG(T)
    data = run!(scheme, truncrank(16), maxiter(25))
    @test free_energy(data, 1.0; scalefactor = 4.0) ≈ f_bench rtol = 1.0e-2
end

# === LoopTNR ===
@testset "LoopTNR - Gross-Neveu Model" begin
    scheme = LoopTNR(T)
    data = run!(scheme, truncrank(8), maxiter(10))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-3
end

# === c4vCTM ===
@testset "c4vCTM - Gross-Neveu Model" begin
    # Use the Gross-Neveu model, but symmetrize the tensor such that the conditions of C4vCTM are satisfied
    T_flipped = gross_neveu_start(0, 0, 0)
    T_unflipped = permute(flip(T_flipped, (1, 2); inv = true), ((), (3, 4, 2, 1)))
    T_unflipped_C4v = TNRKit.symmetrize_C4v(T_unflipped)
    T_flipped_C4v = permute(flip(T_unflipped_C4v, (3, 4); inv = false), ((4, 3), (1, 2)))

    # Check symmetries

    β = 1.0

    # Calculate the free energy using c4vCTM
    data_c4vCTM = run!(c4vCTM(T_flipped_C4v), truncrank(8), maxiter(10))
    free_energy_c4vCTM = -data_c4vCTM / β

    schemes = [TRG, BTRG, HOTRG, ATRG, LoopTNR]
    for scheme in schemes
        data = run!(scheme(T_flipped_C4v), truncrank(8), maxiter(10))
        scalefactor = scheme ∈ [HOTRG, ATRG] ? 4.0 : 2.0
        @test free_energy_c4vCTM ≈ free_energy(data, β; scalefactor) rtol = 1.0e-9
    end
end
