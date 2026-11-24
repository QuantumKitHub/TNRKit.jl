using Test
using TNRKit
using TensorKit
using StableRNGs

T = classical_ising()

# GILTTNR
@testset "GILTTNR - Ising Model" begin
    @info "GILTTNR ising free energy and CFT data"
    scheme = GILTTNR(T; ε = 1.0e-5)
    data = run!(scheme, truncrank(16), maxiter(14))

    cft = sort(CFTData(scheme; shape = [1, 1, 0]).scaling_dimensions[2:end]; by = abs) .|> real
    @test cft[1] ≈ ising_cft_exact[1] rtol = 1.0e-2
    @test cft[2] ≈ ising_cft_exact[2] rtol = 2.0e-2

    # continue the same flow to converge the free energy
    append!(data, run!(scheme, truncrank(16), maxiter(10); finalize_beginning = false))
    @test free_energy(data, ising_βc; scalefactor = 2.0) ≈ f_onsager rtol = 1.0e-6

    @info "Gilt removes the loops of a corner double line tensor"
    d = 3
    rng = StableRNG(1)
    C = [randn(rng, d, d) for _ in 1:4]
    T_cdl = zeros(d, d, d, d, d, d, d, d) # legs (W S N E), each leg split into (lower/left, upper/right) lines
    for (w1, w2, s1, s2, n1, n2, e1, e2) in Iterators.product(fill(1:d, 8)...)
        T_cdl[w1, w2, s1, s2, n1, n2, e1, e2] = C[1][w1, s1] * C[2][s2, e1] * C[3][n2, e2] * C[4][w2, n1]
    end
    T_cdl = TensorMap(reshape(T_cdl, d^2, d^2, d^2, d^2), ℂ^(d^2) ⊗ ℂ^(d^2) ← ℂ^(d^2) ⊗ ℂ^(d^2))
    T1, T2 = TNRKit.gilt_plaquette(T_cdl, T_cdl, 1.0e-7, 20)
    @test all(i -> dim(space(T1, i)) == d && dim(space(T2, i)) == d, 1:4)
end
