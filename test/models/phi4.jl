using Test
using TNRKit
using TensorKit
using TensorKitSectors

model_temp_answer_string_phi4 = [
    (phi4_real(Trivial, 10, -1.0, 1.0), -1.0, 0.4241912271276211, "Real φ⁴ model with no symmetry"), # This is an approximation!
    (phi4_real(10, -1.0, 1.0), -1.0, 0.4232381701937374, "Real φ⁴ model with ℤ₂ symmetry"), # This is an approximation!
    (phi4_complex(Trivial, 6, -1.0, 1.0), -1.0, 0.7583605364656325, "Complex φ⁴ model with no symmetry"), # This is an approximation!
    (phi4_complex(U1Irrep, 6, -1.0, 1.0), -1.0, 0.7673189874157453, "Complex φ⁴ model with U(1) symmetry"), # This is an approximation!
    (phi4_complex(Z2Irrep ⊠ Z2Irrep, 6, -1.0, 1.0), -1.0, 0.7665677554973079, "Complex φ⁴ model with ℤ₂ × ℤ₂ symmetry"), # This is an approximation!
    (phi4_complex(CU1Irrep, 6, -1.0, 1.0), -1.0, 0.7673190424140416, "Complex φ⁴ model with CU(1) or O(2) symmetry"), # This is an approximation!
]

for (model, temp, answer, description) in model_temp_answer_string_phi4
    @testset "$(description)" begin
        scheme = TRG(model)
        data = run!(scheme, truncrank(16), maxiter(25))
        @test free_energy(data, temp) ≈ answer rtol = 1.0e-3
    end
end
