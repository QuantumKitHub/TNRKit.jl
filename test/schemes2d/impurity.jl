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

@testset "ImpurityHOTRG - Constructors" begin
    β = 0.3
    T = classical_ising(Trivial, β)
    T1 = classical_ising_impurity(β)
    T2 = classical_ising_impurity(β; order = 2)

    # First order, single species, general and legacy constructors
    @test keys(ImpurityHOTRG(T, T1).T_imp) == Set([(1, 0)])
    @test keys(ImpurityHOTRG(T, [T1, T2, T1]).T_imp) == Set([(1, 0), (2, 0), (3, 0)])
    @test keys(ImpurityHOTRG(T, Dict((1, 0) => T1, (0, 1) => T1, (1, 1) => T2)).T_imp) == Set([(1, 0), (0, 1), (1, 1)])
    @test keys(ImpurityHOTRG(T, T1, T1, T2).T_imp) == Set([(1, 0), (0, 1), (1, 1)])

    # Invalid input
    @test_throws AssertionError ImpurityHOTRG(T, Dict{NTuple{2, Int}, typeof(T)}())   # no impurities
    @test_throws AssertionError ImpurityHOTRG(T, Dict((0, 0) => T1))                  # (0, 0) is reserved
    @test_throws AssertionError ImpurityHOTRG(T, Dict((1, 0) => T1, (3, 0) => T1))    # (2, 0) missing
    @test_throws AssertionError ImpurityHOTRG(T, Dict((1, 0) => T1, (1, 1) => T1))    # (0, 1) missing
    @test_throws AssertionError ImpurityHOTRG(T, Dict((-1, 1) => T1))                 # negative order
    T_wrong = TensorMap(randn(3, 3, 3, 3), ℂ^3 ⊗ ℂ^3 ← ℂ^3 ⊗ ℂ^3)
    @test_throws AssertionError ImpurityHOTRG(T_wrong, T_wrong, T_wrong, T1)          # different spaces
end

@testset "ImpurityHOTRG - Single species vs legacy second order" begin
    # ⟨m²⟩ computed with the single species orders (1, 0), (2, 0) must equal
    # the one computed with the legacy two species orders (1, 0), (0, 1), (1, 1)
    β = 0.4
    T = classical_ising(Trivial, β)
    T1 = classical_ising_impurity(β)
    T2 = classical_ising_impurity(β; order = 2)

    data_legacy = run!(ImpurityHOTRG(T, T1, T1, T2), truncrank(12), maxiter(15))
    data_single = run!(ImpurityHOTRG(T, [T1, T2]), truncrank(12), maxiter(15))

    @test expectation_value(data_legacy, (1, 1)) ≈ expectation_value(data_single, 2) rtol = 1.0e-6
    @test expectation_value(data_legacy, (1, 0)) ≈ expectation_value(data_single, 1) rtol = 1.0e-6
    @test getindex.(data_legacy, 1) ≈ getindex.(data_single, 1)
end

@testset "ImpurityHOTRG - Magnetisation" begin
    # High temperature limit (<m^2> -> 0)
    β = 0.2

    T = classical_ising(Trivial, β)
    T_imp_order1_1 = classical_ising_impurity(β)
    T_imp_order2 = classical_ising(Trivial, β)

    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)

    data = run!(scheme, truncrank(12), maxiter(25))

    m2_highT = expectation_value(data, (1, 1))
    @test m2_highT ≈ 0.0 atol = 1.0e-8

    # Low temperature limit (<m^2> -> 1)
    β = 1.0

    T = classical_ising(Trivial, β)
    T_imp_order1_1 = classical_ising_impurity(β)
    T_imp_order2 = classical_ising(Trivial, β)

    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)

    data = run!(scheme, truncrank(12), maxiter(25))

    m2_lowT = expectation_value(data, (1, 1))
    @test m2_lowT ≈ 0.9985520252066132 rtol = 1.0e-3
end

@testset "ImpurityHOTRG - Higher order moments and Binder ratio (Ising)" begin
    impurities(β) = [classical_ising_impurity(β; order = k) for k in 1:4]

    # High temperature: odd moments vanish, U4 -> 3, UL -> 0
    β = 0.2
    data = run!(ImpurityHOTRG(classical_ising(Trivial, β), impurities(β)), truncrank(12), maxiter(25))
    @test expectation_value(data, 1) ≈ 0.0 atol = 1.0e-8
    @test expectation_value(data, 3) ≈ 0.0 atol = 1.0e-8
    @test binder_ratio(data) ≈ 3.0 rtol = 1.0e-8
    @test binder_cumulant(data) ≈ 0.0 atol = 1.0e-8

    # Low temperature: ⟨m²⟩, ⟨m⁴⟩ -> m₀², m₀⁴ (Yang), U4 -> 1, UL -> 2/3
    β = 1.0
    m0 = (1 - sinh(2β)^-4)^(1 / 8)
    data = run!(ImpurityHOTRG(classical_ising(Trivial, β), impurities(β)), truncrank(12), maxiter(25))
    @test expectation_value(data, 2) ≈ m0^2 rtol = 1.0e-5
    @test expectation_value(data, 4) ≈ m0^4 rtol = 1.0e-5
    @test binder_ratio(data) ≈ 1.0 rtol = 1.0e-5
    @test binder_cumulant(data) ≈ 2 / 3 rtol = 1.0e-5

    # Critical point: U4 ≈ 1.1679
    β = ising_βc
    U4_critical = 1.1679
    data = run!(ImpurityHOTRG(classical_ising(Trivial, β), impurities(β)), truncrank(20), maxiter(5))
    @test binder_ratio(data; iteration = 5) ≈ U4_critical rtol = 1.0e-2
    @test binder_cumulant(data; iteration = 5) ≈ 1 - U4_critical / 3 rtol = 1.0e-2
end

@testset "ImpurityHOTRG - Observable functions" begin
    β = 0.5
    T_imps = [classical_ising_impurity(β; order = k) for k in 1:2]
    data = run!(ImpurityHOTRG(classical_ising(Trivial, β), T_imps), truncrank(8), maxiter(5))

    # iteration keyword
    @test length(expectation_value(data, 2; iteration = :)) == length(data)
    @test expectation_value(data, 2; iteration = :)[end] == expectation_value(data, 2)
    @test expectation_value(data, 2) == expectation_value(data, (2, 0))
    @test expectation_value(data, 0) ≈ 1.0 rtol = 1.0e-8

    # Wrong orders
    @test_throws ArgumentError expectation_value(data, 3)
    @test_throws ArgumentError expectation_value(data, (0, 1))
    @test_throws ArgumentError binder_ratio(data)
end

@testset "ImpurityHOTRG - Potts model (two species)" begin
    q = 3
    impurities(β) = Dict((a, b) => classical_potts_impurity(q, β, a, b) for a in 0:2, b in 0:2 if (a, b) != (0, 0))

    # High temperature: ⟨|m|²⟩ -> 0, U4 -> 2, UL -> 0
    β = 0.1
    data = run!(ImpurityHOTRG(classical_potts_impurity(q, β, 0, 0), impurities(β)), truncrank(12), maxiter(25))
    @test expectation_value(data, (1, 1)) ≈ 0.0 atol = 1.0e-7
    @test binder_ratio(data) ≈ 2.0 rtol = 1.0e-4
    @test binder_cumulant(data) ≈ 0.0 atol = 1.0e-4

    # Low temperature: ⟨|m|²⟩ -> 1, U4 -> 1, UL -> 1/2
    β = 4.0
    data = run!(ImpurityHOTRG(classical_potts_impurity(q, β, 0, 0), impurities(β)), truncrank(12), maxiter(25))
    @test expectation_value(data, (1, 1)) ≈ 1.0 rtol = 1.0e-4
    @test binder_ratio(data) ≈ 1.0 rtol = 1.0e-6
    @test binder_cumulant(data) ≈ 0.5 rtol = 1.0e-6
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
