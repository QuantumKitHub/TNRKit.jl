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

    data = run!(scheme, truncrank(12), maxiter(25))

    order_para = expectation_value(data, (1, 1))
    @test order_para ≈ 0.0 atol = 1.0e-6

    # Ordered phase
    μ0 = -2.0

    Tpure = phi4_real(Trivial, 10, μ0, λ)
    T_imp1 = phi4_real_imp1(10, μ0, λ)
    T_imp2 = phi4_real_imp2(10, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp1, T_imp1, T_imp2)

    data = run!(scheme, truncrank(12), maxiter(25))

    order_para = expectation_value(data, (1, 1))
    @test order_para ≈ 1.5316631237138074 rtol = 1.0e-3
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

    data = run!(scheme, truncrank(12), maxiter(25))

    order_para = expectation_value(data, (1, 1))
    @test real(order_para) ≈ 0.0 atol = 1.0e-6

    # Quasi long range order phase
    λ = 1.0
    μ0 = -2.0

    Tpure = phi4_complex(Trivial, 6, μ0, λ; T = ComplexF64)
    T_imp11 = phi4_complex_impϕ(6, μ0, λ)
    T_imp12 = phi4_complex_impϕdag(6, μ0, λ)
    T_imp2 = phi4_complex_impϕ2(6, μ0, λ)

    scheme = ImpurityHOTRG(Tpure, T_imp11, T_imp12, T_imp2)

    data = run!(scheme, truncrank(12), maxiter(25))

    order_para = expectation_value(data, (1, 1))
    @test real(order_para) ≈ 1.1690143852848816 rtol = 1.0e-2

    # ABS PHI IMPURITY
    # Quasi long range order phase
    λ = 1.0
    μ0 = -2.0

    Tpure, T_ϕ, T_ϕdag, T_ϕabs, T_ϕ2 = phi4_complex_all(6, μ0, λ)
    T_abs = phi4_complex_impϕabs(6, μ0, λ)

    # phi4_complex_all and phi4_complex_impϕabs give the same tensor (up to an SVD gauge)
    @test norm(T_abs) ≈ norm(T_ϕabs)

    scheme = ImpurityHOTRG(Tpure, T_abs) # first order only

    data = run!(scheme, truncrank(12), maxiter(25))

    order_para = expectation_value(data, 1)
    @test real(order_para) ≈ 1.296354266089897 rtol = 1.0e-2

end

# Impurity tensors of arbitrary order
@testset "Impurity tensors of arbitrary order" begin
    # Ising: σ^order, σ² = 1
    β = 0.3
    @test classical_ising_impurity(β; order = 2) ≈ classical_ising(Trivial, β)
    @test classical_ising_impurity(β; order = 3, h = 0.1) ≈ classical_ising_impurity(β; h = 0.1)
    @test_throws AssertionError classical_ising_impurity(β; order = -1)

    # Real ϕ⁴: ϕ^n
    K, μ0, λ = 6, -1.0, 1.0
    @test phi4_real_imp(K, μ0, λ, 0) ≈ phi4_real(Trivial, K, μ0, λ)
    @test phi4_real_imp(K, μ0, λ, 1, 0.1) ≈ phi4_real_imp1(K, μ0, λ, 0.1)
    @test phi4_real_imp(K, μ0, λ, 2) ≈ phi4_real_imp2(K, μ0, λ)
    @test_throws ArgumentError phi4_real_imp(K, μ0, λ, -1)

    # Complex ϕ⁴: ϕ^a (ϕ†)^b
    K = 4
    T_all = phi4_complex_all(K, μ0, λ)
    @test phi4_complex_imp(K, μ0, λ, 0, 0) ≈ phi4_complex(Trivial, K, μ0, λ; T = ComplexF64)
    @test phi4_complex_imp(K, μ0, λ, 1, 0) ≈ T_all[2] ≈ phi4_complex_impϕ(K, μ0, λ)
    @test phi4_complex_imp(K, μ0, λ, 0, 1) ≈ T_all[3] ≈ phi4_complex_impϕdag(K, μ0, λ)
    @test phi4_complex_imp(K, μ0, λ, 1, 1) ≈ T_all[5] ≈ phi4_complex_impϕ2(K, μ0, λ)
    @test_throws ArgumentError phi4_complex_imp(K, μ0, λ, -1, 0)
end

# Real ϕ⁴ moments and Binder ratio
@testset "Real φ⁴ model - Binder cumulant" begin
    K, λ = 10, 1.0
    for (μ0, UL) in ((0.0, 0.0), (-2.0, 2 / 3)) # disordered and ordered phase
        Tpure = phi4_real(Trivial, K, μ0, λ)
        T_imps = [phi4_real_imp(K, μ0, λ, n) for n in 1:4]
        data = run!(ImpurityHOTRG(Tpure, T_imps), truncrank(12), maxiter(25))
        @test binder_cumulant(data) ≈ UL atol = 1.0e-6
    end
end
