using Test
using TNRKit
using TensorKit

function cc_finalize!(scheme::LoopTNR)
    n = finalize!(scheme)
    τ, c = extract_tau_and_c(scheme.TA, scheme.TB)
    @info "c = $c, τ = $τ"
    return (n, τ, c)
end

@testset "Quantum Ising chain: $stack_alg stacking" for stack_alg in (:exponential, :linear)
    trunc_stack = truncerror(; rtol = 1.0e-9) & truncrank(16)
    if stack_alg == :exponential
        nfold = 7
        dt = 1 / (2^nfold)
        T = quantum_ising_chain(Float64, Z2Irrep, dt; J = 1.0, g = 1.0)
        T = vertical_stack_exp(T, nfold, trunc_stack)
    else
        n = 100
        dt = 1 / n
        T = quantum_ising_chain(Float64, Z2Irrep, dt; J = 1.0, g = 1.0)
        T = vertical_stack_linear(T, n, trunc_stack)
    end
    scheme = LoopTNR(T)
    elt = scalartype(T)
    finalizer = Finalizer(cc_finalize!, Tuple{elt, complex(elt), elt})
    data = run!(scheme, truncrank(16), maxiter(16), finalizer; finalize_beginning = false)
    @test last(data)[3] ≈ 0.5 atol = 1.0e-2
end
