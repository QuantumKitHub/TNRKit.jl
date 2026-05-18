using Test
using TNRKit
using TensorKit

function _thermal_zn2_gu_wen_x1(β, Lz; χttnr = 12, χbtrg = 16, btrg_steps = 14)
    scheme = ThermalTNR(ZN_gauge_theory_dual(2, β))
    run!(scheme, truncrank(χttnr), maxiter(Lz - 1); verbosity = 0)

    @tensor effective_tensor[-1 -2 -3 -4] := scheme.T[1, 1][p p; -1 -2 -3 -4]
    btrg = BTRG(permute(effective_tensor, ((1, 2), (3, 4))))
    ratios = run!(
        btrg, truncrank(χbtrg), maxiter(btrg_steps), guwenratio_Finalizer;
        finalize_beginning = false, verbosity = 0,
    )

    x1, _ = last(ratios)
    return x1
end

@testset "ThermalTNR Z₂ gauge dual Gu-Wen ratio" begin
    # Table 2 of arXiv:2602.13124 gives the dual-spin estimates of βc(Lz).
    # The dual representation reverses the phases: β < βc has X₁ ≈ 2, β > βc has X₁ ≈ 1.
    for (Lz, βc) in ((2, 0.65605), (4, 0.731065))
        x1_below = _thermal_zn2_gu_wen_x1(βc - 0.1, Lz)
        x1_above = _thermal_zn2_gu_wen_x1(βc + 0.1, Lz)

        @test x1_below ≈ 2 rtol = 1.0e-2
        @test x1_above ≈ 1 rtol = 5.0e-2
    end
end
