using Test
using TNRKit
using TensorKit

model_temp_answer_string_3d = [
    (classical_ising_3D(Trivial), ising_βc_3D, -3.508, "3D Ising model with no symmetry"), # This is an approximation!
    (classical_ising_3D(), ising_βc_3D, -3.508, "3D Ising model with ℤ₂ symmetry"), # This is an approximation!
]

for (model, temp, answer, description) in model_temp_answer_string_3d
    @testset "$(description)" begin
        scheme = HOTRG_3D(model)
        data = run!(scheme, truncrank(8), maxiter(25))
        @test free_energy(data, temp; scalefactor = 8.0) ≈ answer rtol = 1.0e-3
    end
end
