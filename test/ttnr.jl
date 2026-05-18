@testset "ThermalTNR exports" begin
    @test !(:TNO in names(TNRKit))
    @test !(:TNOTensor in names(TNRKit))
end

@testset "ThermalTNR construction" begin
    local_tensor = randn(ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    unitcell = [copy(local_tensor) for _ in 1:2, _ in 1:2]
    scheme = ThermalTNR(unitcell)

    @test size(scheme.T) == (2, 2)
    @test scheme.T[2, 1] ≈ local_tensor
    @test !contains(sprint(show, scheme), "TNO")
    @test_throws ArgumentError ThermalTNR(reshape(typeof(local_tensor)[], 0, 1))
end

@testset "ThermalTNR apply!" begin
    local_tensor = randn(ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    top = [copy(local_tensor) for _ in 1:2, _ in 1:2]
    bottom = [copy(local_tensor) for _ in 1:2, _ in 1:2]
    scheme_top = ThermalTNR(top)
    scheme_bottom = ThermalTNR(bottom)
    merged = apply!(scheme_top, scheme_bottom, truncrank(8))

    @test merged isa ThermalTNR
    @test size(merged.T) == (2, 2)
    @test space(merged.T[1, 1], 1) == space(top[1, 1], 1)
    @test space(merged.T[1, 1], 2) == space(bottom[1, 1], 2)

    bad_tensor = randn(ℂ^2 ⊗ (ℂ^4)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')
    @test_throws ArgumentError apply!(
        ThermalTNR([bad_tensor;;]), ThermalTNR([local_tensor;;]), truncrank(8)
    )
end

@testset "ThermalTNR finalize!" begin
    local_tensor = randn(ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')


    scheme = ThermalTNR([copy(local_tensor) for _ in 1:2, _ in 1:2])
    n = finalize!(scheme)

    @test isfinite(n)
    @test n > 0
    @test n ≈ norm(@tensor local_tensor[1 1; 2 3 2 3])
    @test all(norm(@tensor scheme.T[i, j][1 1; 2 3 2 3]) ≈ 1 for i in 1:2, j in 1:2)
end

@testset "ThermalTNR run!" begin
    local_tensor = randn(ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    scheme = ThermalTNR([copy(local_tensor);;])
    layer = [copy(local_tensor);;]
    data = run!(scheme, layer, truncrank(8), maxiter(1))

    @test data isa Vector{Float64}
    @test length(data) == 2
    @test all(isfinite, data)
    @test all(n -> n > 0, data)
    @test norm(@tensor scheme.T[1, 1][1 1; 2 3 2 3]) ≈ 1
end
