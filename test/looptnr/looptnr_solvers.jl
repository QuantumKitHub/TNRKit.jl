using Test
using TNRKit
using TensorKit

T = classical_ising()

@testset "LoopTNR - Ising Model - Dense Solver - NNR" begin
    @info "LoopTNR ising free energy"
    scheme = LoopTNR(T)

    loop_condition = LoopParameters(
        sweeping = maxiter(5) & convcrit(1.0e-9, (steps, cost) -> abs(cost[end])),
        truncentanglement = trunctol(atol = 1.0e-12),
        nuclear_norm = true
    )

    data = run!(
        scheme, truncrank(8), maxiter(25), loop_condition
    )

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-6
end

@testset "LoopTNR - Ising Model - Krylov Solver" begin
    @info "LoopTNR ising free energy"
    scheme = LoopTNR(T)

    loop_condition = LoopParameters(
        sweeping = maxiter(5) & convcrit(1.0e-9, (steps, cost) -> abs(cost[end])),
        truncentanglement = trunctol(atol = 1.0e-12),
        krylov = true
    )

    data = run!(
        scheme, truncrank(8), maxiter(25), loop_condition
    )

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-6
end

@testset "LoopTNR - Ising Model - Krylov Solver- NNR" begin
    @info "LoopTNR ising free energy"
    scheme = LoopTNR(T)

    loop_condition = LoopParameters(
        sweeping = maxiter(5) & convcrit(1.0e-9, (steps, cost) -> abs(cost[end])),
        truncentanglement = trunctol(atol = 1.0e-12),
        krylov = true,
        nuclear_norm = true
    )

    data = run!(
        scheme, truncrank(8), maxiter(25), loop_condition
    )

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-6
end


@testset "LoopTNR - Initialization with 2 x 2 unit cell" begin
    loop_condition = LoopParameters(
        sweeping = maxiter(5) & convcrit(1.0e-12, (steps, cost) -> abs(cost[end]))
    )
    trunc = truncrank(8)
    truncentanglement = trunctol(atol = 1.0e-12)
    entanglement_criterion = maxiter(100)
    scheme = LoopTNR(fill(T, (2, 2)); trunc, loop_condition)
    data = run!(
        scheme, truncrank(8), maxiter(25), loop_condition
    )
    @test free_energy(data, ising_βc; initial_size = 2) ≈ f_onsager rtol = 1.0e-6
end
