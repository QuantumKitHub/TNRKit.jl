using Test
using TNRKit
using TensorKit

# c6vCTM_triangular
@testset "c6vCTM_triangular - Ising Model" begin
    for sym in [Trivial, Z2Irrep]
        for projectors in [:twothirds :full]
            for conditioning in [true false]
                T_flipped = classical_ising_triangular(sym, ising_βc_triangular)

                scheme = c6vCTM_triangular(T_flipped)
                lz = run!(scheme, truncrank(20), maxiter(100); projectors, conditioning)

                fs = lz * -1 / ising_βc_triangular
                @test fs ≈ f_onsager_triangular rtol = 1.0e-4
            end
        end
    end
end

# CTM_triangular
@testset "CTM_triangular - Ising Model" begin
    for sym in [Trivial, Z2Irrep]
        for projectors in [:twothirds :full]
            for conditioning in [true false]
                T_flipped = classical_ising_triangular(sym, ising_βc_triangular)

                scheme = CTM_triangular(T_flipped)
                lz = run!(scheme, truncrank(20), maxiter(100); projectors, conditioning)

                fs = lz * -1 / ising_βc_triangular
                @test fs ≈ f_onsager_triangular rtol = 1.0e-4
            end
        end
    end
end
