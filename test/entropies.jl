println("-----------------------")
println(" Testing all entropies ")
println("-----------------------")

@testset "von Neumann entropy of a TensorMap" begin
    M = TensorMap([0.5 0 0; 0 0.5 0; 0 0 0.5], ℂ^3 ← ℂ^3)
    S_von, spec = VN_entropy(M)
    @test isapprox(S_von, log(3))
    @test isapprox(spec.data, [1.0, 1.0, 1.0])

    N = TensorMap([10.0 0.0 0; 0.0 0.0 0; 0 0 0], ℂ^3 ← ℂ^3)
    S_von_N, spec_N = VN_entropy(N)
    @test isapprox(S_von_N, 0.0)
    @test isapprox(spec_N.data, [0.0, 0.0, 1.0])
end

@testset "NNR-TNR can reduce loop entropy" begin
    T = classical_ising()

    scheme_loop = LoopTNR(T)
    scheme_nnr = LoopTNR(T)

    loop_condition = LoopParameters()
    nnr_condition = LoopParameters(nuclear_norm = true)

    run!(scheme_loop, truncrank(16), maxiter(15), loop_condition; verbosity = 1)
    run!(scheme_nnr, truncrank(16), maxiter(15), nnr_condition; verbosity = 1)

    entropies_loop, _, entropies_rad_loop, _ = loop_entropy(scheme_loop)
    entropies_nnr, _, entropies_rad_nnr, _ = loop_entropy(scheme_nnr)

    @test all(abs.(entropies_nnr) .< abs.(entropies_loop))
    @test all(abs.(entropies_rad_nnr) .< abs.(entropies_rad_loop))

    @info "Loop entropies (LoopTNR): $entropies_loop"
    @info "Loop entropies (NNR-TNR): $entropies_nnr"
    @info "Radial entropies (LoopTNR): $entropies_rad_loop"
    @info "Radial entropies (NNR-TNR): $entropies_rad_nnr"
end
