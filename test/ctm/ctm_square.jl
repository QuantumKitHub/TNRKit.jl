using Test
using TNRKit
using TensorKit

T = classical_ising()

# ctm
@testset "CTM - Ising Model" begin
    @info "CTM ising free energy"
    scheme = CTM(T)

    lz = run!(scheme, truncrank(32), maxiter(256))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 1.0e-6
end

# Sublattice CTM
@testset "Sublattice_CTM - Ising Model" begin
    @info "Sublattice_CTM ising free energy"
    scheme = Sublattice_CTM(T, T)

    lz = run!(scheme, truncrank(32), maxiter(256))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 1.0e-6
end

# ctm_TRG
@testset "ctm_TRG - Ising Model" begin
    @info "ctm_TRG ising free energy"
    scheme = ctm_TRG(T, 8)
    lz = run!(scheme, truncrank(8), maxiter(25))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 7.0e-6
end

# ctm_HOTRG
@testset "ctm_HOTRG - Ising Model" begin
    @info "ctm_HOTRG ising free energy"
    scheme = ctm_HOTRG(T, 8)
    lz = run!(scheme, truncrank(8), maxiter(25))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 2.0e-5
end

# c4vCTM
@testset "c4vCTM - Ising Model" begin
    @info "c4vCTM ising free energy"
    scheme = c4vCTM(T)
    lz = run!(scheme, truncrank(24), trivial_convcrit(1.0e-9); verbosity = 1)

    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 6.0e-8
end

# rCTM
@testset "rCTM - Ising Model" begin
    @info "rCTM ising free energy"
    scheme = rCTM(T)
    lz = run!(scheme, truncrank(24), trivial_convcrit(1.0e-9); verbosity = 1)

    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 6.0e-8
end
