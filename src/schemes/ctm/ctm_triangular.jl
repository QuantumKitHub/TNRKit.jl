# Based on
# https://arxiv.org/pdf/2510.04907

function run!(
        scheme::CTM_triangular, trunc::MatrixAlgebraKit.TruncationStrategy, criterion::stopcrit;
        projectors = :twothirds,
        conditioning = true,
        verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        Ss_prev = [id(domain(scheme.C[dir])) for dir in 1:6]
        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), ε = $(ε)"

            Ss = step!(scheme, trunc; projectors, conditioning)

            ε = calculate_error(Ss, Ss_prev)
            Ss_prev = Ss

            steps += 1
            crit = criterion(steps, ε)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function step!(scheme::CTM_triangular, trunc; projectors = :twothirds, conditioning = true)
    Pas, Pbs, Ss = calculate_projectors(scheme, trunc, projectors)

    renormalize_corners!(scheme, Pas, Pbs)
    normalize_corners!(scheme)

    Ẽas, Ẽbs, Ẽastr, Ẽbstr = semi_renormalize(scheme, Pas, Pbs, trunc)
    Qas, Qbs = build_matrix_second_projectors(scheme, Ẽas, Ẽbs, Ẽastr, Ẽbstr, trunc; conditioning)

    renormalize_edges!(scheme, Ẽas, Ẽbs, Qas, Qbs)
    normalize_edges!(scheme)
    return Ss
end

function calculate_projectors(scheme::Union{CTM_triangular, c6vCTM_triangular}, trunc, projectors)
    if projectors == :full
        return calculate_full_projectors(scheme, trunc)
    elseif projectors == :twothirds
        return calculate_twothirds_projectors(scheme, trunc)
    else
        @error "projectors = $projectors not defined"
    end
end

function calculate_twothirds_projectors(scheme, trunc)
    Pas = []
    Pbs = []
    Ss = []
    for dir in 1:6
        ρL = build_double_corner_matrix_triangular(scheme, mod1(dir - 1, 6))
        ρR = build_double_corner_matrix_triangular(scheme, mod1(dir + 1, 6))
        @tensor ρρ[-1 -2; -3 -4] := ρL[-1 -2; 1 2] * flip(ρR, 2; inv = false)[1 2; -3 -4]
        ρρ /= norm(ρρ)

        U, S, V = svd_trunc(ρρ; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())

        Pb = ρR * V' * pseudopow(S, -1 / 2)
        Pa = pseudopow(S, -1 / 2) * U' * ρL
        push!(Pas, Pa)
        push!(Pbs, Pb)
        push!(Ss, S)
    end
    return Pas, Pbs, Ss
end

function calculate_full_projectors(scheme, trunc)
    Pas = []
    Pbs = []
    Ss = []
    for dir in 1:6
        ρL = build_double_corner_matrix_triangular(scheme, mod1(dir - 1, 6))
        ρR = build_double_corner_matrix_triangular(scheme, mod1(dir + 1, 6))
        ρ̄ = build_double_corner_matrix_triangular(scheme, mod1(dir + 3, 6))
        ρ̄ /= norm(ρ̄)
        Ū, S̄, V̄ᴴ = svd_trunc(ρ̄; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
        ρ̄ᴿ = Ū * sqrt(S̄)
        ρ̄ᴸ = sqrt(S̄) * V̄ᴴ
        @tensor ρρ[-1; -2] := ρ̄ᴸ[-1; 1 2] * flip(ρL, 2; inv = false)[1 2; 3 4] * flip(ρR, 2; inv = false)[3 4; 5 6] * flip(ρ̄ᴿ, 2; inv = false)[5 6; -2]
        ρρ /= norm(ρρ)


        U, S, Vᴴ = svd_trunc(ρρ; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())

        @tensor Pb[-1 -2; -3] := ρR[-1 -2; 1 2] * flip(ρ̄ᴿ, 2)[1 2; 3] * Vᴴ'[3; 4] * pseudopow(S, -1 / 2)[4; -3]
        @tensor Pa[-1; -2 -3] := pseudopow(S, -1 / 2)[-1; 1] * U'[1; 2] * ρ̄ᴸ[2; 3 4] * flip(ρL, 2)[3 4; -2 -3]
        push!(Pas, Pa)
        push!(Pbs, Pb)
        push!(Ss, S)
    end
    return Pas, Pbs, Ss
end

function renormalize_corners!(scheme, Pas, Pbs)
    for dir in 1:6
        @tensor opt = true scheme.C[dir][-1 -2; -3] := scheme.C[dir][1 3; 6] * scheme.Ea[mod1(dir - 1, 6)][4 2; 1] * scheme.Eb[dir][6 7; 8] *
            flip(scheme.T, (3, 4, 5); inv = false)[3 7 9 -2 5 2] * Pas[mod1(dir - 1, 6)][-1; 4 5] * Pbs[dir][8 9; -3]
    end
    return
end

function network_value(scheme::Union{CTM_triangular, c6vCTM_triangular})
    nw_corners = _contract_corners(scheme)
    nw_full = _contract_site_large(scheme)
    nw_0 = _contract_edges_0(scheme)
    nw_60 = _contract_edges_60(scheme)
    nw_120 = _contract_edges_120(scheme)
    return (nw_full * nw_corners^2 / (nw_0 * nw_60 * nw_120))^(1 / 3)
end

function lnz(scheme::Union{CTM_triangular, c6vCTM_triangular})
    return real(log(network_value(scheme)))
end

function build_double_corner_matrix_triangular(scheme::CTM_triangular, dir::Int)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.C[dir][6 5; 1] * scheme.C[mod1(dir + 1, 6)][1 3; 2] *
        scheme.Ea[mod1(dir - 1, 6)][-1 7; 6] * scheme.Eb[mod1(dir + 1, 6)][2 4; -3] * scheme.T[5 3 4 -4 -2 7]
    return mat
end

function semi_renormalize(scheme::CTM_triangular, Pas, Pbs, trunc)
    Ẽas = []
    Ẽbs = []
    Ẽastr = []
    Ẽbstr = []
    for dir in 1:6
        @tensor opt = true mat[-1 -2; -3 -4] := Pas[dir][-1; 1 2] * Pbs[dir][6 7; -3] *
            scheme.Ea[dir][1 3; 4] * scheme.Eb[dir][4 5; 6] * flip(scheme.T, (3, 4, 5, 6); inv = false)[3 5 7 -4 -2 2]
        mat /= norm(mat)

        U, S, V = svd_trunc(mat; trunc = trunctol(; atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())

        Ẽb = U * sqrt(S)
        Ẽa = permute(sqrt(S) * V, ((1, 3), (2,)))

        Utr, Str, Vtr = svd_trunc(mat; trunc = trunc & trunctol(; atol = 1.0e-20), alg = MatrixAlgebraKit.LAPACK_QRIteration())
        Ẽbtr = Utr * sqrt(Str)
        Ẽatr = permute(sqrt(Str) * Vtr, ((1, 3), (2,)))

        push!(Ẽas, Ẽa)
        push!(Ẽbs, Ẽb)
        push!(Ẽastr, Ẽatr)
        push!(Ẽbstr, Ẽbtr)
    end
    return Ẽas, Ẽbs, Ẽastr, Ẽbstr
end

function build_matrix_second_projectors(scheme::CTM_triangular, Ẽas, Ẽbs, Ẽastr, Ẽbstr, trunc; conditioning = true)
    Qas = []
    Qbs = []
    for dir in 1:6
        @tensor opt = true σL[-1 -2; -3] := scheme.C[dir][1 2; 8] * scheme.C[mod1(dir - 1, 6)][4 3; 1] * scheme.C[mod1(dir - 2, 6)][6 5; 4] *
            scheme.T[2 9 -2 7 5 3] * Ẽbs[dir][8 9; -3] * Ẽastr[mod1(dir - 3, 6)][-1 7; 6]
        @tensor opt = true σR[-1; -2 -3] := scheme.C[mod1(dir + 1, 6)][8 2; 1] * scheme.C[mod1(dir + 2, 6)][1 3; 4] * scheme.C[mod1(dir + 3, 6)][4 5; 6] *
            scheme.T[9 2 3 5 7 -3] * Ẽbstr[mod1(dir + 3, 6)][6 7; -2] * Ẽas[dir][-1 9; 8]

        if conditioning
            σL /= norm(σL)
            σR /= norm(σR)
            UL, SL, VLᴴ = svd_trunc(σL; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
            UR, SR, VRᴴ = svd_trunc(σR; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())

            FLU = sqrt(SL) * VLᴴ
            FRU = UR * sqrt(SR)

            mat = FLU * FRU
            mat /= norm(mat)
            WU, SU, QUᴴ = svd_trunc(mat; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())

            Qa = pseudopow(SU, -1 / 2) * WU' * FLU
            Qb = FRU * QUᴴ' * pseudopow(SU, -1 / 2)
        else
            mat = σL * σR
            mat /= norm(mat)
            U, S, V = svd_trunc(mat; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
            Qa = pseudopow(S, -1 / 2) * U' * σL
            Qb = σR * V' * pseudopow(S, -1 / 2)
        end
        push!(Qas, Qa)
        push!(Qbs, Qb)
    end
    return Qas, Qbs
end

function renormalize_edges!(scheme, Ẽas, Ẽbs, Qas, Qbs)
    for dir in 1:6
        @tensor scheme.Eb[dir][-1 -2; -3] := Ẽbs[dir][-1 -2; 1] * Qbs[dir][1; -3]
        @tensor scheme.Ea[dir][-1 -2; -3] := Qas[dir][-1; 1] * Ẽas[dir][1 -2; -3]
    end
    return scheme
end

function normalize_corners!(scheme)
    for dir in 1:6
        scheme.C[dir] /= norm(scheme.C[dir])
    end
    return scheme
end

function normalize_edges!(scheme)
    for dir in 1:6
        scheme.Ea[dir] /= norm(scheme.Ea[dir])
        scheme.Eb[dir] /= norm(scheme.Eb[dir])
    end
    return scheme
end

function calculate_error(Ss, Ss_prev)
    ε = Inf
    for (S, S_prev) in zip(Ss, Ss_prev)
        if space(S) == space(S_prev)
            ε = norm(S^4 - S_prev^4)
        else
            return Inf
        end
    end
    return ε
end

function Base.show(io::IO, scheme::CTM_triangular)
    println(io, "CTM_triangular - Corner Transfer Matrix for triangular lattices")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * Ea: $(summary(scheme.Ea))")
    println(io, "  * Eb: $(summary(scheme.Eb))")
    return nothing
end

# Expectation_values

function _contract_edges_0(scheme::CTM_triangular)
    return @tensor opt = true flip(scheme.T, 3; inv = false)[DL120 DL60 DL0 DL300 DL240 DL180] * scheme.T[DR120 DR60 DR0 DR300 DR240 DL0] *
        scheme.C[1][χNW DL120; χNa] * scheme.C[2][χNb DR60; χNE] * scheme.C[3][χNE DR0; χSE] *
        scheme.C[4][χSE DR300; χSa] * scheme.C[5][χSb DL240; χSW] * scheme.C[6][χSW DL180; χNW] *
        scheme.Eb[1][χNa DL60; χNC] * scheme.Ea[1][χNC DR120; χNb] *
        scheme.Eb[4][χSa DR240; χSC] * scheme.Ea[4][χSC DL300; χSb]
end

function _contract_edges_60(scheme::CTM_triangular)
    return @tensor opt = true flip(scheme.T, 5; inv = false)[DTR120 DTR60 DTR0 DTR300 DBL60 DTR180] * scheme.T[DBL120 DBL60 DBL0 DBL300 DBL240 DBL180] *
        scheme.C[1][χNWb DTR120; χN] * scheme.C[2][χN DTR60; χNE] * scheme.C[3][χNE DTR0; χSEa] *
        scheme.C[4][χSEb DBL300; χS] * scheme.C[5][χS DBL240; χSW] * scheme.C[6][χSW DBL180; χNWa] *
        scheme.Eb[3][χSEa DTR300; χSEC] * scheme.Ea[3][χSEC DBL0; χSEb] *
        scheme.Eb[6][χNWa DBL120; χNWC] * scheme.Ea[6][χNWC DTR180; χNWb]
end

function _contract_edges_120(scheme::CTM_triangular)
    return @tensor opt = true flip(scheme.T, 4; inv = false)[DTL120 DTL60 DTL0 DTL300 DTL240 DTL180] * scheme.T[DTL300 DBR60 DBR0 DBR300 DBR240 DBR180] *
        scheme.C[1][χNW DTL120; χN] * scheme.C[2][χN DTL60; χNEa] * scheme.C[3][χNEb DBR0; χSE] *
        scheme.C[4][χSE DBR300; χS] * scheme.C[5][χS DBR240; χSWa] * scheme.C[6][χSWb DTL180; χNW] *
        scheme.Eb[2][χNEa DTL0; χNEC] * scheme.Ea[2][χNEC DBR60; χNEb] *
        scheme.Eb[5][χSWa DBR180; χSWC] * scheme.Ea[5][χSWC DTL240; χSWb]
end

function _contract_site_large(scheme::CTM_triangular)
    return @tensor opt = true flip(scheme.T, 5; inv = false)[DNW120 DNW60 DNW0 DNW300 DW60 DNW180] * flip(scheme.T, 6; inv = false)[DNE120 DNE60 DNE0 DNE300 DNE240 DNW0] *
        flip(scheme.T, 1; inv = false)[DNE300 DE60 DE0 DE300 DE240 DE180] * flip(scheme.T, 2; inv = false)[DSE120 DE240 DSE0 DSE300 DSE240 DSE180] * flip(scheme.T, 3; inv = false)[DSW120 DSW60 DSE180 DSW300 DSW240 DSW180] *
        flip(scheme.T, 4; inv = false)[DW120 DW60 DW0 DSW120 DW240 DW180] * flip(scheme.T, (1, 2, 3, 4, 5, 6); inv = false)[DNW300 DNE240 DE180 DSE120 DSW60 DW0] *
        scheme.C[1][χNWa DNW120; χNb] * scheme.Eb[1][χNb DNW60; χNC] * scheme.Ea[1][χNC DNE120; χNa] *
        scheme.C[2][χNa DNE60; χNEb] * scheme.Eb[2][χNEb DNE0; χNEC] * scheme.Ea[2][χNEC DE60; χNEa] *
        scheme.C[3][χNEa DE0; χSEb] * scheme.Eb[3][χSEb DE300; χSEC] * scheme.Ea[3][χSEC DSE0; χSEa] *
        scheme.C[4][χSEa DSE300; χSb] * scheme.Eb[4][χSb DSE240; χSC] * scheme.Ea[4][χSC DSW300; χSa] *
        scheme.C[5][χSa DSW240; χSWb] * scheme.Eb[5][χSWb DSW180; χSWC] * scheme.Ea[5][χSWC DW240; χSWa] *
        scheme.C[6][χSWa DW180; χNWb] * scheme.Eb[6][χNWb DW120; χNWC] * scheme.Ea[6][χNWC DNW180; χNWa]
end

function _contract_corners(scheme::CTM_triangular)
    return @tensor opt = true scheme.C[1][χNW D120; χN] * scheme.C[2][χN D60; χNE] * scheme.C[3][χNE D0; χSE] *
        scheme.C[4][χSE D300; χS] * scheme.C[5][χS D240; χSW] * scheme.C[6][χSW D180; χNW] *
        scheme.T[D120 D60 D0; D300 D240 D180]
end
