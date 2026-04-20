using StableRNGs
@testset "Honeycomb schemes - Ising Model" begin
    for sym in [Trivial, Z2Irrep]
        for alg in [:CTM_honeycomb, :c3vCTM_honeycomb]
            T_flipped = classical_ising_honeycomb(sym, ising_βc_honeycomb; T = ComplexF64)
            scheme = eval(alg)(T_flipped)
            lz = run!(scheme, truncrank(20), convcrit(1.0e-4, (steps, data) -> data) & maxiter(300); verbosity = 1)

            fs = lz * -1 / ising_βc_honeycomb
            @test fs ≈ f_onsager_honeycomb rtol = 1.0e-2
        end
    end
end

# Test honeycomb CTM by converting it to CTM on a square lattice
@testset "Honeycomb CTM C3 - Random Model" begin
    rng = StableRNG(1234)
    for sym in [Trivial, Z2Irrep]
        for alg in [:CTM_honeycomb, :c3vCTM_honeycomb]
            A = zeros(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)
            A.data .= rand(rng, length(A.data))
            A /= norm(A)

            if alg == :c3vCTM_honeycomb
                scheme = eval(alg)(A; symmetrize = true)
            else
                scheme = eval(alg)(A)
            end
            lz_honeycomb = run!(scheme, truncrank(20), convcrit(1.0e-4, (steps, data) -> data) & maxiter(300); verbosity = 1)

            @tensor pf_square[-4 -3; -1 -2] := A[-1 -2 1] * flip(A, [1 2 3])[-3 -4 1]
            scheme_square = CTM(pf_square)
            lz_square = run!(scheme, truncrank(20), convcrit(1.0e-4, (steps, data) -> data) & maxiter(300); verbosity = 1)

            @test lz_square ≈ lz_honeycomb rtol = 1.0e-3
        end
    end
end

# Test CTM_honeycomb for A ≠ B by converting it to CTM on a square lattice
@testset "Honeycomb CTM - Random Model" begin
    rng = StableRNG(1234)
    for sym in [Trivial, Z2Irrep]
        A = zeros(ComplexF64, ℂ^2 ⊗ ℂ^3, ℂ^4)
        B = zeros(ComplexF64, ℂ^2 ⊗ ℂ^3, ℂ^4)
        A.data .= rand(rng, length(A.data))
        B.data .= rand(rng, length(B.data))
        A /= norm(A)
        B /= norm(B)

        @test_throws ArgumentError c3vCTM_honeycomb(A)
        scheme = CTM_honeycomb(A; B)
        lz_honeycomb = run!(scheme, truncrank(40), convcrit(-Inf, (steps, data) -> data) & maxiter(500); verbosity = 1)

        @tensor pf_square[-4 -3; -1 -2] := A[-1 -2 1] * flip(B, [1 2 3])[-3 -4 1]
        scheme_square = CTM(pf_square)
        lz_square = run!(scheme, truncrank(40), convcrit(1.0e-14, (steps, data) -> data) & maxiter(500); verbosity = 1)

        @test lz_square ≈ lz_honeycomb rtol = 1.0e-2
    end
end

@testset "Rotations for honeycomb lattice" begin
    rng = StableRNG(1234)
    A_flipped = zeros(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)
    A = permute(flip(A_flipped, [1 2]; inv = true), ((), (3, 2, 1)))
    A.data .= rand(rng, length(A.data))

    @test TNRKit.rotl120_pf_honeycomb(A, 3) ≈ A
    @test TNRKit.rotl120_pf_honeycomb(A, 1) ≈ TNRKit.rotl120_pf_honeycomb(A)
    @test TNRKit.is_C3_symmetric(TNRKit.symmetrize_C3_honeycomb(A))
end
