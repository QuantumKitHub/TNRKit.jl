using Test
using TNRKit
using TensorKit

T_3D = classical_ising_3D()
const f_benchmark3D = ising_3D_free_energy_htse()

# ATRG_3D
@testset "ATRG_3D - Ising Model" begin
    @info "ATRG_3D ising free energy"
    scheme = ATRG_3D(T_3D)
    data = run!(scheme, truncrank(12), maxiter(25))
    fs = free_energy(data, ising_βc_3D; scalefactor = 8.0)
    @info "Calculated f = $(fs)."
    @test fs ≈ f_benchmark3D rtol = 5.0e-3
end

# HOTRG_3D
@testset "HOTRG_3D - Ising Model" begin
    @info "HOTRG_3D ising free energy"
    scheme = HOTRG_3D(T_3D)
    data = run!(scheme, truncrank(8), maxiter(25))
    fs = free_energy(data, ising_βc_3D; scalefactor = 8.0)
    @info "Calculated f = $(fs)."
    @test fs ≈ f_benchmark3D rtol = 1.0e-3
end

@testset "HOTRG_3D - Projector for fermions" begin
    @info "HOTRG_3D projectors for fermions"
    Vphy = Vect[FermionParity](0 => 2, 1 => 2)
    Vvir = Vect[FermionParity](0 => 2, 1 => 2)
    for _ in 1:4 # multiple trials
        Aspace = (Vphy ⊗ Vphy' ← Vvir ⊗ Vvir ⊗ Vvir' ⊗ Vvir')
        A1 = randn(ComplexF64, Aspace)
        A2 = randn(ComplexF64, Aspace)
        for MM in [TNRKit._get_MMdag_3d(A1, A2), TNRKit._get_MdagM_3d(A1, A2)]
            @test isposdef(MM)
        end
    end
end
