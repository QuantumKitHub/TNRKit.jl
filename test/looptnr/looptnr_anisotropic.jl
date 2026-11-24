using Test
using TNRKit
using TensorKit

const Jx_aniso, Jy_aniso = 1.0, 0.6
const βc_aniso = ising_anisotropic_βc(Jx_aniso, Jy_aniso)
const T_aniso = classical_ising(Z2Irrep, βc_aniso; Jx = Jx_aniso, Jy = Jy_aniso)
const f_aniso_exact = f_onsager_anisotropic(βc_aniso, Jx_aniso, Jy_aniso)
const τ_aniso_exact = sinh(2 * βc_aniso * Jx_aniso)

"""
Normalize the tensors, return the normalization factor and elementary modular parameter
"""
function tau_finalize!(scheme::LoopTNR)
    n = finalize!(scheme)
    τ0, c = extract_tau_and_c(scheme.TA, scheme.TB; fast = false)
    return (n, τ0)
end

@testset "LoopTNR - Anisotropic Ising Model" begin
    @info "Anisotropy: Jx = $(Jx_aniso), Jy = $(Jy_aniso)"
    @info "LoopTNR anisotropic ising free energy"
    scheme = LoopTNR(T_aniso)

    loop_condition = LoopParameters(
        sweeping = maxiter(5) & convcrit(1.0e-9, (steps, cost) -> abs(cost[end])),
        truncentanglement = trunctol(atol = 1.0e-12)
    )
    elt = scalartype(T_aniso)
    finalizer = Finalizer(tau_finalize!, Tuple{elt, complex(elt)})
    data = run!(scheme, truncrank(8), maxiter(25), loop_condition, finalizer)

    ns = map(Base.Fix2(getindex, 1), data)
    @test free_energy(ns, βc_aniso) ≈ f_aniso_exact rtol = 1.0e-6

    @info "LoopTNR τ → (1 + τ) / (1 - τ)"
    f_looptnr(τ) = (1 + τ) / (1 - τ)
    τs = map(Base.Fix2(getindex, 2), data)
    for n in 5:8
        @test τs[n + 1] ≈ f_looptnr(τs[n]) rtol = 2.0e-2
        @info "* verified for step $(n - 1) → $n"
    end

    @info "Theory value of τ for anisotropic Ising"
    for n in (4, 8, 12)
        # n + 1 due to finalizing the initial tensor
        τ = τs[n + 1]
        @test real(τ) ≈ 0 atol = 1.0e-3
        @test imag(τ) ≈ τ_aniso_exact rtol = 2.0e-3
        @info "* τ = $τ ≈ $(τ_aniso_exact)im verified for step $n"
    end

    @info "LoopTNR anisotropic ising CFT data"
    scheme = LoopTNR(T_aniso)
    run!(scheme, truncrank(12), maxiter(10))

    # use fast tau algorithm below
    for shape in ("[1, 4, 1]", "[√2, 2√2, 0]")
        cft = CFTData(scheme; shape = eval(Meta.parse(shape)))
        d_σ = real(cft.scaling_dimensions[Z2Irrep(1)][1])
        d_ε = real(cft.scaling_dimensions[Z2Irrep(0)][2])
        @info "Shape $shape: Δ(σ) = $d_σ, Δ(ε) = $d_ε, c = $(cft.central_charge)"
        @test d_σ ≈ ising_cft_exact[1] rtol = 5.0e-4
        @test d_ε ≈ ising_cft_exact[2] rtol = 5.0e-4
        @test cft.central_charge ≈ 0.5 rtol = 5.0e-3
        # conformal spins: only [1, 4, 1] (x=1) resolves them
        if shape == "[1, 4, 1]"
            sd = cft.scaling_dimensions
            # σ (Z₂ odd, first state): s = 0
            s_σ = -imag(sd[Z2Irrep(1)][1])
            @test abs(s_σ) < 1.0e-6
            # ε (Z₂ even, second state): s = 0
            s_ε = -imag(sd[Z2Irrep(0)][2])
            @test abs(s_ε) < 1.0e-6
            # all spins in the low-lying spectrum should be integer
            for sector in keys(sd)
                for v in sd[sector]
                    Δ, s = real(v), -imag(v)
                    Δ > 2.5 && break  # check only low-lying states
                    @test isapprox(s, round(s); atol = 1.0e-4)
                end
            end
            @info "Conformal spins are integer-valued."
        end
    end

    for shape in ("[1, 8, 1]", "[4/√10, 2√10, 2/√10]")
        cft = CFTData(
            scheme; shape = eval(Meta.parse(shape)), trunc = truncrank(16),
            truncentanglement = trunctol(atol = 1.0e-10)
        )
        d_σ = real(cft.scaling_dimensions[Z2Irrep(1)][1])
        d_ε = real(cft.scaling_dimensions[Z2Irrep(0)][2])
        @info "Shape $shape:  Δ(σ) = $d_σ,  Δ(ε) = $d_ε,  c = $(cft.central_charge)"
        @test d_σ ≈ ising_cft_exact[1] rtol = 2.0e-3
        @test d_ε ≈ ising_cft_exact[2] rtol = 2.0e-3
        @test cft.central_charge ≈ 0.5 rtol = 1.0e-2
    end

    @info "LoopTNR anisotropic ising ground state degeneracy"
    T1 = classical_ising(βc_aniso - 0.01; Jx = Jx_aniso, Jy = Jy_aniso)
    scheme = LoopTNR(T1)
    run!(scheme, truncrank(12), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 1 rtol = 1.0e-2
    @test X1 ≈ 1.0 rtol = 1.0e-2
    @test X2 ≈ 1.0 rtol = 1.0e-2

    T2 = classical_ising(βc_aniso + 0.01; Jx = Jx_aniso, Jy = Jy_aniso)
    scheme = LoopTNR(T2)
    run!(scheme, truncrank(12), maxiter(20))
    gsd = ground_state_degeneracy(scheme)
    X1, X2 = gu_wen_ratio(scheme)
    @test gsd ≈ 2 rtol = 1.0e-2
    @test X1 ≈ 2.0 rtol = 1.0e-2
    @test X2 ≈ 2.0 rtol = 1.0e-2
end
