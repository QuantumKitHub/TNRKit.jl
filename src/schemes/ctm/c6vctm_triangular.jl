function run!(
        scheme::c6vCTM_triangular, trunc::TruncationStrategy, criterion::stopcrit;
        projectors = :twothirds,
        conditioning = true,
        verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        S_prev = id(domain(scheme.C))

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), ε = $(ε)"

            S = step!(scheme, trunc; projectors, conditioning)

            if space(S) == space(S_prev)
                ε = norm(S^4 - S_prev^4)
            end
            S_prev = S

            steps += 1
            crit = criterion(steps, ε)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function step!(scheme::c6vCTM_triangular, trunc; projectors = :twothirds, conditioning = true)
    Pa, Pb, S = calculate_projectors(scheme, trunc, projectors)

    renormalize_corners!(scheme, Pa, Pb)
    scheme.C /= norm(scheme.C)

    Ẽa, Ẽb, Ẽatr, Ẽbtr = semi_renormalize(scheme, Pa, Pb, trunc)
    Qa, Qb = build_matrix_second_projector(scheme, Ẽa, Ẽb, Ẽatr, Ẽbtr; conditioning)

    renormalize_edges!(scheme, Ẽa, Ẽb, Qa, Qb)
    scheme.Ea /= norm(scheme.Ea)
    scheme.Eb /= norm(scheme.Eb)
    return S
end

function calculate_twothirds_projectors(scheme::c6vCTM_triangular, trunc)
    ρ = build_double_corner_matrix_triangular(scheme)
    @tensor ρρ[-1 -2; -3 -4] := ρ[-1 -2; 1 2] * flip(ρ, 2; inv = false)[1 2; -3 -4]
    ρρ /= norm(ρρ)

    U, S, V = svd_trunc(ρρ; trunc = trunc & trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())

    Pb = ρ * V' * pseudopow(S, -1 / 2)
    Pa = pseudopow(S, -1 / 2) * U' * ρ
    return Pa, Pb, S
end

function calculate_full_projectors(scheme::c6vCTM_triangular, trunc)
    ρ = build_double_corner_matrix_triangular(scheme)
    ρ /= norm(ρ)
    Ū, S̄, V̄ᴴ = svd_trunc(ρ; trunc = trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())
    ρ̄ᴿ = Ū * sqrt(S̄)
    ρ̄ᴸ = sqrt(S̄) * V̄ᴴ
    @tensor ρρ[-1; -2] := ρ̄ᴸ[-1; 1 2] * flip(ρ, 2; inv = false)[1 2; 3 4] * flip(ρ, 2; inv = false)[3 4; 5 6] * flip(ρ̄ᴿ, 2; inv = false)[5 6; -2]
    ρρ /= norm(ρρ)

    U, S, Vᴴ = svd_trunc(ρρ; trunc = trunc & trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())

    @tensor Pb[-1 -2; -3] := ρ[-1 -2; 1 2] * flip(ρ̄ᴿ, 2)[1 2; 3] * Vᴴ'[3; 4] * pseudopow(S, -1 / 2)[4; -3]
    @tensor Pa[-1; -2 -3] := pseudopow(S, -1 / 2)[-1; 1] * U'[1; 2] * ρ̄ᴸ[2; 3 4] * flip(ρ, 2)[3 4; -2 -3]
    return Pa, Pb, S
end

function renormalize_corners!(scheme::c6vCTM_triangular, Pa, Pb)
    @tensor opt = true scheme.C[-1 -2; -3] := scheme.C[1 3; 6] * scheme.Ea[4 2; 1] * scheme.Eb[6 7; 8] *
        flip(scheme.T, (3, 4, 5); inv = false)[3 7 9 -2 5 2] * Pa[-1; 4 5] * Pb[8 9; -3]
    return scheme
end

function build_double_corner_matrix_triangular(scheme::c6vCTM_triangular)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.C[1 3; 2] * scheme.C[6 5; 1] *
        scheme.Ea[-1 7; 6] * scheme.Eb[2 4; -3] * scheme.T[3 4 -4 -2 7 5]
    return mat
end

function semi_renormalize(scheme::c6vCTM_triangular, Pa, Pb, trunc)
    @tensor opt = true mat[-1 -2; -3 -4] := Pa[-1; 1 2] * Pb[6 7; -3] *
        scheme.Ea[1 3; 4] * scheme.Eb[4 5; 6] * flip(scheme.T, (3, 4, 5, 6); inv = false)[3 5 7 -4 -2 2]
    U, S, V = svd_trunc(mat; trunc = trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())
    Ẽb = U * sqrt(S)
    Ẽa = permute(sqrt(S) * V, ((1, 3), (2,)))

    Utr, Str, Vtr = svd_trunc(mat; trunc = trunc & trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())
    Ẽbtr = Utr * sqrt(Str)
    Ẽatr = permute(sqrt(Str) * Vtr, ((1, 3), (2,)))

    return Ẽa, Ẽb, Ẽatr, Ẽbtr
end

function build_matrix_second_projector(scheme::c6vCTM_triangular, Ẽa, Ẽb, Ẽatr, Ẽbtr; conditioning = true)
    @tensor opt = true σL[-1 -2; -3] := scheme.C[1 2; 8] * scheme.C[4 3; 1] * scheme.C[6 5; 4] *
        scheme.T[2 9 -2 7 5 3] * Ẽb[8 9; -3] * Ẽatr[-1 7; 6]
    @tensor opt = true σR[-1; -2 -3] := scheme.C[8 2; 1] * scheme.C[1 3; 4] * scheme.C[4 5; 6] *
        scheme.T[9 2 3 5 7 -3] * Ẽbtr[6 7; -2] * Ẽa[-1 9; 8]

    if conditioning
        σL /= norm(σL)
        σR /= norm(σR)
        UL, SL, VLᴴ = svd_trunc(σL; trunc = trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())
        UR, SR, VRᴴ = svd_trunc(σR; trunc = trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())

        FLU = sqrt(SL) * VLᴴ
        FRU = UR * sqrt(SR)

        mat = FLU * FRU
        mat /= norm(mat)
        WU, SU, QUᴴ = svd_trunc(mat; trunc = trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())

        Qa = pseudopow(SU, -1 / 2) * WU' * FLU
        Qb = FRU * QUᴴ' * pseudopow(SU, -1 / 2)
    else
        mat = σL * σR
        mat /= norm(mat)
        U, S, V = svd_trunc(mat; trunc = trunctol(atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())
        Qa = pseudopow(S, -1 / 2) * U' * σL
        Qb = σR * V' * pseudopow(S, -1 / 2)
    end
    return Qa, Qb
end

function renormalize_edges!(scheme::c6vCTM_triangular, Ẽa, Ẽb, Qa, Qb)
    @tensor scheme.Eb[-1 -2; -3] := Ẽb[-1 -2; 1] * Qb[1; -3]
    @tensor scheme.Ea[-1 -2; -3] := Qa[-1; 1] * Ẽa[1 -2; -3]
    return scheme
end

function Base.show(io::IO, scheme::c6vCTM_triangular)
    println(io, "c6vCTM_triangular - C6v symmetric Corner Transfer Matrix for triangular lattices")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * Ea: $(summary(scheme.Ea))")
    println(io, "  * Eb: $(summary(scheme.Eb))")
    return nothing
end

# Expectation_values

function _contract_edges_0(scheme::c6vCTM_triangular)
    return @tensor opt = true flip(scheme.T, 3; inv = false)[DL120 DL60 DL0 DL300 DL240 DL180] * scheme.T[DR120 DR60 DR0 DR300 DR240 DL0] *
        scheme.C[χNW DL120; χNa] * scheme.C[χNb DR60; χNE] * scheme.C[χNE DR0; χSE] *
        scheme.C[χSE DR300; χSa] * scheme.C[χSb DL240; χSW] * scheme.C[χSW DL180; χNW] *
        scheme.Eb[χNa DL60; χNC] * scheme.Ea[χNC DR120; χNb] *
        scheme.Eb[χSa DR240; χSC] * scheme.Ea[χSC DL300; χSb]
end

function _contract_edges_60(scheme::c6vCTM_triangular)
    return @tensor opt = true flip(scheme.T, 5; inv = false)[DTR120 DTR60 DTR0 DTR300 DBL60 DTR180] * scheme.T[DBL120 DBL60 DBL0 DBL300 DBL240 DBL180] *
        scheme.C[χNWb DTR120; χN] * scheme.C[χN DTR60; χNE] * scheme.C[χNE DTR0; χSEa] *
        scheme.C[χSEb DBL300; χS] * scheme.C[χS DBL240; χSW] * scheme.C[χSW DBL180; χNWa] *
        scheme.Eb[χSEa DTR300; χSEC] * scheme.Ea[χSEC DBL0; χSEb] *
        scheme.Eb[χNWa DBL120; χNWC] * scheme.Ea[χNWC DTR180; χNWb]
end

function _contract_edges_120(scheme::c6vCTM_triangular)
    return @tensor opt = true flip(scheme.T, 4; inv = false)[DTL120 DTL60 DTL0 DTL300 DTL240 DTL180] * scheme.T[DTL300 DBR60 DBR0 DBR300 DBR240 DBR180] *
        scheme.C[χNW DTL120; χN] * scheme.C[χN DTL60; χNEa] * scheme.C[χNEb DBR0; χSE] *
        scheme.C[χSE DBR300; χS] * scheme.C[χS DBR240; χSWa] * scheme.C[χSWb DTL180; χNW] *
        scheme.Eb[χNEa DTL0; χNEC] * scheme.Ea[χNEC DBR60; χNEb] *
        scheme.Eb[χSWa DBR180; χSWC] * scheme.Ea[χSWC DTL240; χSWb]
end

function _contract_site_large(scheme::c6vCTM_triangular)
    return @tensor opt = true flip(scheme.T, 5; inv = false)[DNW120 DNW60 DNW0 DNW300 DW60 DNW180] * flip(scheme.T, 6; inv = false)[DNE120 DNE60 DNE0 DNE300 DNE240 DNW0] *
        flip(scheme.T, 1; inv = false)[DNE300 DE60 DE0 DE300 DE240 DE180] * flip(scheme.T, 2; inv = false)[DSE120 DE240 DSE0 DSE300 DSE240 DSE180] * flip(scheme.T, 3; inv = false)[DSW120 DSW60 DSE180 DSW300 DSW240 DSW180] *
        flip(scheme.T, 4; inv = false)[DW120 DW60 DW0 DSW120 DW240 DW180] * flip(scheme.T, (1, 2, 3, 4, 5, 6); inv = false)[DNW300 DNE240 DE180 DSE120 DSW60 DW0] *
        scheme.C[χNWa DNW120; χNb] * scheme.Eb[χNb DNW60; χNC] * scheme.Ea[χNC DNE120; χNa] *
        scheme.C[χNa DNE60; χNEb] * scheme.Eb[χNEb DNE0; χNEC] * scheme.Ea[χNEC DE60; χNEa] *
        scheme.C[χNEa DE0; χSEb] * scheme.Eb[χSEb DE300; χSEC] * scheme.Ea[χSEC DSE0; χSEa] *
        scheme.C[χSEa DSE300; χSb] * scheme.Eb[χSb DSE240; χSC] * scheme.Ea[χSC DSW300; χSa] *
        scheme.C[χSa DSW240; χSWb] * scheme.Eb[χSWb DSW180; χSWC] * scheme.Ea[χSWC DW240; χSWa] *
        scheme.C[χSWa DW180; χNWb] * scheme.Eb[χNWb DW120; χNWC] * scheme.Ea[χNWC DNW180; χNWa]
end

function _contract_corners(scheme::c6vCTM_triangular)
    return @tensor opt = true scheme.C[χNW D120; χN] * scheme.C[χN D60; χNE] * scheme.C[χNE D0; χSE] *
        scheme.C[χSE D300; χS] * scheme.C[χS D240; χSW] * scheme.C[χSW D180; χNW] *
        scheme.T[D120 D60 D0; D300 D240 D180]
end
