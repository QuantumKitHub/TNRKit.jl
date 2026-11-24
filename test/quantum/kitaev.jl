using Test
using TNRKit
using TensorKit

function cc_finalize!(scheme::LoopTNR)
    n = finalize!(scheme)
    τ, c = extract_tau_and_c(scheme.TA, scheme.TB)
    @info "c = $c, τ = $τ"
    return (n, τ, c)
end

@testset "Quantum Kitaev chain: Ising case" begin
    trunc_stack = truncerror(; rtol = 1.0e-9) & truncrank(16)
    nfold = 7
    dt = 1 / (2^nfold)
    T = kitaev_chain(Float64, Trivial, dt; t = 1.0, Δ = 1.0, V = 0.0, μ = 2.0)
    T = vertical_stack_exp(T, nfold, trunc_stack)

    @info "CFT data from TRG"
    scheme = TRG(T)
    run!(scheme, truncrank(24), maxiter(8))
    cft = CFTData(scheme; shape = [1, 1, 0])
    c = cft.central_charge
    sd = cft.scaling_dimensions
    d_1 = real(sd[(:NS, FermionParity(0))][2])
    d_f = real(sd[(:NS, FermionParity(1))][1])
    d_e = real(sd[(:R, FermionParity(0))][1])
    d_m = real(sd[(:R, FermionParity(1))][1])
    @info "  shape [1, 1, 0]:\nΔ(1)=$d_1, Δ(f)=$d_f, Δ(e)=$d_e, Δ(m)=$d_m, c=$c"
    @test d_1 ≈ 1 rtol = 2.0e-2
    @test d_f ≈ 1 / 2 rtol = 2.0e-2
    @test d_e ≈ 1 / 8 rtol = 2.0e-2
    @test d_m ≈ 1 / 8 rtol = 2.0e-2
    @test c ≈ 0.5 rtol = 2.0e-2

    @info "CFT data from LoopTNR"
    scheme = LoopTNR(T)
    elt = scalartype(T)
    finalizer = Finalizer(cc_finalize!, Tuple{elt, complex(elt), elt})
    data = run!(scheme, truncrank(16), maxiter(8), finalizer; finalize_beginning = false)
    @test last(data)[3] ≈ 0.5 atol = 1.0e-2
    for shape in ([√2, 2√2, 0], [1, 4, 1], [1, 8, 1], [4 / √10, 2√10, 2 / √10])
        cft = CFTData(scheme; shape = shape)
        c = cft.central_charge
        sd = cft.scaling_dimensions
        d_1 = real(sd[(:NS, FermionParity(0))][2])
        d_f = real(sd[(:NS, FermionParity(1))][1])
        d_e = real(sd[(:R, FermionParity(0))][1])
        d_m = real(sd[(:R, FermionParity(1))][1])
        @info "  shape $shape:\nΔ(1)=$d_1, Δ(f)=$d_f, Δ(e)=$d_e, Δ(m)=$d_m, c=$c"
        @test d_1 ≈ 1 rtol = 1.0e-2
        @test d_f ≈ 1 / 2 rtol = 1.0e-2
        @test d_e ≈ 1 / 8 rtol = 1.0e-2
        @test d_m ≈ 1 / 8 rtol = 1.0e-2
        @test c ≈ 0.5 rtol = 1.0e-2

        shape[end] == 0 && continue
        for sect in (
                (:NS, FermionParity(0)), (:NS, FermionParity(1)),
                (:R, FermionParity(0)), (:R, FermionParity(1)),
            )
            for v in sd[sect]
                Δ, s = real(v), -imag(v)
                Δ > 2.5 && break
                s′ = (sect == (:NS, FermionParity(1))) ? s - 0.5 : s
                @test isapprox(s′, round(s′); atol = 1.0e-2)
            end
            @info "  Conformal spins are integer (for 1, e, m) or half-integer (for f)."
        end
    end
end
