using Test
using TNRKit
using TensorKit

# Test for impurity tensors
# Real phi^4
@testset "Real φ⁴ model - Impure" begin
    # Disordered phase
    λ = 1.0
    μ0 = 0.0

    Tpure = phi4_real(Trivial, 10, μ0, λ)
    T_imp1 = phi4_real_imp1(10, μ0, λ)
    T_imp2 = phi4_real_imp2(10, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp1, T_imp1, T_imp2)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][4] / data[end][1]
    @test order_para ≈ 0.0 atol = 1.0e-3

    # Ordered phase
    μ0 = -2.0

    Tpure = phi4_real(Trivial, 10, μ0, λ)
    T_imp1 = phi4_real_imp1(10, μ0, λ)
    T_imp2 = phi4_real_imp2(10, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp1, T_imp1, T_imp2)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][4] / data[end][1]
    @test order_para ≈ 1.5317112652447245 rtol = 1.0e-3
end

# Complex phi^4
@testset "Complex φ⁴ model - Impure" begin
    # Disordered phase
    λ = 1.0
    μ0 = 0.0

    Tpure = phi4_complex(Trivial, 6, μ0, λ; T = ComplexF64)
    T_imp11 = phi4_complex_impϕ(6, μ0, λ)
    T_imp12 = phi4_complex_impϕdag(6, μ0, λ)
    T_imp2 = phi4_complex_impϕ2(6, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp11, T_imp12, T_imp2)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][4] / data[end][1]
    @test order_para ≈ 0.0 atol = 1.0e-7

    # Quasi long range order phase
    λ = 1.0
    μ0 = -2.0

    Tpure = phi4_complex(Trivial, 6, μ0, λ; T = ComplexF64)
    T_imp11 = phi4_complex_impϕ(6, μ0, λ)
    T_imp12 = phi4_complex_impϕdag(6, μ0, λ)
    T_imp2 = phi4_complex_impϕ2(6, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp11, T_imp12, T_imp2)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][4] / data[end][1]
    @test order_para ≈ 1.17802947935296 rtol = 1.0e-2

    # ABS PHI IMPURITY
    # Quasi long range order phase
    λ = 1.0
    μ0 = -2.0

    Tpure, T_ϕ, T_ϕdag, T_ϕabs, T_ϕ2 = phi4_complex_all(6, μ0, λ)
    T_abs = phi4_complex_impϕabs(6, μ0, λ)

    # phi4_complex_all and phi4_complex_impϕabs give the same tensor (up to an SVD gauge)
    @test norm(T_abs) ≈ norm(T_ϕabs)

    scheme = ImpurityHOTRG(Tpure, T_abs, Tpure, Tpure)

    data = run!(scheme, truncrank(16), maxiter(25))

    order_para = data[end][2] / data[end][1]
    @test order_para ≈ 1.2957057423957146 rtol = 1.0e-2

end
