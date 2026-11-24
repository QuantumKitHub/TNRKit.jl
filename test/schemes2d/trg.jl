using Test
using TNRKit
using TensorKit

const Jx_aniso, Jy_aniso = 1.0, 0.6
const βc_aniso = ising_anisotropic_βc(Jx_aniso, Jy_aniso)
const T_aniso = classical_ising(Z2Irrep, βc_aniso; Jx = Jx_aniso, Jy = Jy_aniso)
const f_aniso_exact = f_onsager_anisotropic(βc_aniso, Jx_aniso, Jy_aniso)

"""
Normalize the tensor, return the normalization factor and elementary modular parameter
"""
function tau_finalize!(scheme::TRG)
    n = finalize!(scheme)
    τ0, c = extract_tau_and_c(scheme.T; fast = false)
    return (n, τ0)
end

T = classical_ising()

# TRG
@testset "TRG - Anisotropic Ising Model" begin
    @info "Anisotropy: Jx = $(Jx_aniso), Jy = $(Jy_aniso)"
    @info "TRG anisotropic ising free energy"
    scheme = TRG(T_aniso)
    elt = scalartype(T_aniso)
    finalizer = Finalizer(tau_finalize!, Tuple{elt, complex(elt)})
    data = run!(scheme, truncrank(24), maxiter(25), finalizer)

    ns = map(Base.Fix2(getindex, 1), data)
    @test free_energy(ns, βc_aniso) ≈ f_aniso_exact rtol = 2.0e-6

    @info "TRG τ → (τ - 1) / (τ + 1)"
    f_trg(τ) = (τ - 1) / (τ + 1)
    τs = map(Base.Fix2(getindex, 2), data)
    for n in 5:7
        @test τs[n + 1] ≈ f_trg(τs[n]) rtol = 5.0e-2
        @info "* verified for step $(n - 1) → $n"
    end

    @info "TRG anisotropic ising CFT data — shape [1, 1, 0]"
    scheme = TRG(T_aniso)
    run!(scheme, truncrank(24), maxiter(10))
    # use fast tau algorithm below
    cft = CFTData(scheme; shape = [1, 1, 0])
    sd_all = real(cft.scaling_dimensions)
    cft_sorted = sort(sd_all[2:end]; by = abs)

    @test cft_sorted[1] ≈ ising_cft_exact[1] rtol = 2.0e-3
    @test cft_sorted[2] ≈ ising_cft_exact[2] rtol = 2.0e-2
    @info "Obtained scaling dimensions: Δ₁ = $(cft_sorted[1]), Δ₂ = $(cft_sorted[2])"

    @info "TRG anisotropic ising ground state degeneracy"
    T1 = classical_ising(βc_aniso - 0.01; Jx = Jx_aniso, Jy = Jy_aniso)
    scheme = TRG(T1)
    run!(scheme, truncrank(16), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 1 rtol = 1.0e-2
    @test X1 ≈ 1.0 rtol = 1.0e-2
    @test X2 ≈ 1.0 rtol = 1.0e-2

    T2 = classical_ising(βc_aniso + 0.01; Jx = Jx_aniso, Jy = Jy_aniso)
    scheme = TRG(T2)
    run!(scheme, truncrank(16), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 2 rtol = 1.0e-2
    @test X1 ≈ 2.0 rtol = 1.0e-2
    @test X2 ≈ 2.0 rtol = 1.0e-2
end

# BTRG
@testset "BTRG - Ising Model" begin
    @info "BTRG ising free energy"
    scheme = BTRG(T)
    data = run!(scheme, truncrank(24), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 6.0e-8

    @info "BTRG ising CFT data"
    scheme = BTRG(T)
    run!(scheme, truncrank(24), maxiter(10))

    cft = sort(CFTData(scheme; shape = [1, 1, 0]).scaling_dimensions[2:end]; by = abs) .|> real

    @test cft[1] ≈ ising_cft_exact[1] rtol = 3.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 2.0e-2

    @info "BTRG ising ground state degeneracy"
    T1 = classical_ising(ising_βc - 0.01)
    scheme = BTRG(T1)
    run!(scheme, truncrank(16), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 1 rtol = 1.0e-2
    @test X1 ≈ 1.0 rtol = 1.0e-2
    @test X2 ≈ 1.0 rtol = 1.0e-2

    T2 = classical_ising(ising_βc + 0.01)
    scheme = BTRG(T2)
    run!(scheme, truncrank(16), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 2 rtol = 1.0e-2
    @test X1 ≈ 2.0 rtol = 1.0e-2
    @test X2 ≈ 2.0 rtol = 1.0e-2
end
