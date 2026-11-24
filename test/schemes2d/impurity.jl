using Test
using TNRKit
using TensorKit

# ImpurityHOTRG
@testset "ImpurityHOTRG - Ising Model" begin

    T = classical_ising(Trivial)
    T_imp1 = classical_ising_impurity()

    scheme = ImpurityHOTRG(T, T_imp1, T_imp1, T)

    data = run!(scheme, truncrank(16), maxiter(25))

    @test free_energy(getindex.(data, 1), ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 6.0e-7
end

@testset "Impurity HOTRG - Magnetisation" begin
    # High temperature limit (<m^2> -> 0)
    β = 0.2

    T = classical_ising(Trivial, β)
    T_imp_order1_1 = classical_ising_impurity(β)
    T_imp_order2 = classical_ising(Trivial, β)

    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)

    data = run!(scheme, truncrank(8), maxiter(25))

    m2_highT = data[end][4] / data[end][1]
    @test m2_highT ≈ 0.0 atol = 1.0e-14

    # Low temperature limit (<m^2> -> 1)
    β = 1.0

    T = classical_ising(Trivial, β)
    T_imp_order1_1 = classical_ising_impurity(β)
    T_imp_order2 = classical_ising(Trivial, β)

    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)

    data = run!(scheme, truncrank(8), maxiter(25))

    m2_lowT = data[end][4] / data[end][1]
    @test m2_lowT ≈ 1 rtol = 1.0e-2
end

# ImpurityTRG
@testset "ImpurityTRG - Ising Model" begin
    T = classical_ising(Trivial)
    T_imp = classical_ising_impurity()

    scheme = ImpurityTRG(T, T_imp, T, T, T)

    data = run!(scheme, truncrank(24), maxiter(25))

    @test free_energy(getindex.(data, 1), ising_βc) ≈ f_onsager rtol = 2.0e-6
end

@testset "ImpurityTRG - Magnetisation" begin
    # High T
    β = 0.1

    T = classical_ising(Trivial, β)
    T_imp = classical_ising_impurity(β)

    scheme = ImpurityTRG(T, T_imp, T, T, T)

    data = run!(scheme, truncrank(16), maxiter(25))

    m_expection = data[end][2] / data[end][1]
    @test m_expection ≈ 0.0 atol = 1.0e-4

    # Low T
    β = 2

    T = classical_ising(Trivial, β; h = 1.0e-6)
    T_imp = classical_ising_impurity(β; h = 1.0e-6)

    scheme = ImpurityTRG(T, T_imp, T, T, T)

    data = run!(scheme, truncrank(16), maxiter(25))

    m_expection = data[end][2] / data[end][1]
    @test m_expection ≈ 1.0 rtol = 1.0e-4
end

# CorrelationHOTRG
@testset "Correlation HOTRG - Ising Model" begin

    T = classical_ising(Trivial)
    T_imp = classical_ising_impurity()

    scheme = CorrelationHOTRG(T, T_imp, T_imp, 5)

    data = run!(scheme, truncrank(16), maxiter(25))

    @test free_energy(getindex.(data, 1), ising_βc; scalefactor = 4.0, initial_size = 4.0) ≈ f_onsager rtol = 1.0e-4
end

@testset "Correlation HOTRG - Magnetisation Correlation" begin
    # High temperature limit
    β = 0.2

    T = classical_ising(Trivial, β)
    T_imp = classical_ising_impurity(β)

    scheme = CorrelationHOTRG(T, T_imp, T_imp, 5)

    data = run!(scheme, truncrank(16), maxiter(25))

    highT = norm(@tensor scheme.Timp_final[1 2; 2 1]) / norm(@tensor scheme.Tpure[1 2; 2 1])
    @test highT ≈ 7.396177e-6 rtol = 1.0e-5

    # Critical temperature limit
    T = classical_ising(Trivial)
    T_imp = classical_ising_impurity()

    scheme = CorrelationHOTRG(T, T_imp, T_imp, 5)

    data = run!(scheme, truncrank(16), maxiter(25))

    Tc = norm(@tensor scheme.Timp_final[1 2; 2 1]) / norm(@tensor scheme.Tpure[1 2; 2 1])
    @test Tc ≈ 0.2981409 rtol = 1.0e-5

    # Low temperature limit
    β = 3.0

    T = classical_ising(Trivial, β)
    T_imp = classical_ising_impurity(β)

    scheme = CorrelationHOTRG(T, T_imp, T_imp, 5)

    data = run!(scheme, truncrank(16), maxiter(25))

    lowT = norm(@tensor scheme.Timp_final[1 2; 2 1]) / norm(@tensor scheme.Tpure[1 2; 2 1])
    @test lowT ≈ 1 rtol = 1.0e-4
end
