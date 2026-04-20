function run!(
        scheme::CTM_honeycomb, trunc::TruncationStrategy, criterion::stopcrit;
        verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        Ss_prev = [DiagonalTensorMap(id(domain(scheme.C[dir]))) for dir in 1:3]
        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), ε = $(ε)"

            Ss = step!(scheme, trunc)
            ε = error_measure(Ss, Ss_prev)
            Ss_prev = Ss
            steps += 1
            crit = criterion(steps, ε)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function step!(scheme::CTM_honeycomb, trunc)
    # take two half-steps to compensate for the swapping of Ta and Tb in one half-step.
    half_step!(scheme, trunc)
    Ss = half_step!(scheme, trunc)
    return Ss
end

function half_step!(scheme::CTM_honeycomb, trunc)
    projectors, Ss = calculate_projectors(scheme, trunc)
    renormalize_edges!(scheme, projectors)
    return Ss
end

function build_corner_matrices(scheme::CTM_honeycomb)
    @tensor opt = true σ₁[χNW1 DaSW; χNE2 DbSE] := flip(scheme.A, [2 3])[DaNW DaE DaSW] * scheme.B[DbSE DaE DbNE] *
        scheme.Tb[3][χNW1 DaNW; χNW2] * scheme.Ta[1][χNE1 DbNE; χNE2] *
        scheme.C[1][χNW2; χNE1]
    @tensor opt = true σ₂[χNE1 DaNW; χS2 DbW] := flip(scheme.A, [1 3])[DaNW DaE DaSW] * scheme.B[DbSE DbW DaSW] *
        scheme.Tb[1][χNE1 DaE; χNE2] * scheme.Ta[2][χS1 DbSE; χS2] *
        scheme.C[2][χNE2; χS1]
    @tensor opt = true σ₃[χS1 DaE; χNW2 DbNE] := flip(scheme.A, [1 2])[DaNW DaE DaSW] * scheme.B[DaNW DbW DbNE] *
        scheme.Tb[2][χS1 DaSW; χS2] * scheme.Ta[3][χNW1 DbW; χNW2] *
        scheme.C[3][χS2; χNW1]
    return [σ₁, σ₂, σ₃]
end

function calculate_projector_simultaneous(σs, trunc, dir)
    σsshift = circshift(σs, 1 - dir)
    mat = prod(σsshift)
    U, S, Vᴴ = svd_trunc(mat; trunc)
    PL = prod(σsshift[2:end]) * Vᴴ' * pseudopow(S, -1 / 2)
    PR = pseudopow(S, -1 / 2) * U' * σsshift[1]
    return PL, PR, S
end

function calculate_projectors(scheme::CTM_honeycomb, trunc)
    σs = build_corner_matrices(scheme)
    projectors_simul = [calculate_projector_simultaneous(σs, trunc, dir) for dir in 1:3]
    projectors = [projectors_simul[dir][1:2] for dir in 1:3]
    Ss = [projectors_simul[dir][3] for dir in 1:3]
    for dir in 1:3
        C′ = projectors[mod1(dir - 1, 3)][2] * σs[dir] * projectors[dir][1]
        scheme.C[dir] = C′ / norm(C′)
    end
    return projectors, Ss
end

function renormalize_edges!(scheme::CTM_honeycomb, projectors)
    for dir in 1:3
        @tensor opt = true Ta′[χout DSW; χin] := projectors[dir][2][χout; χNE1 DNW] * scheme.Tb[dir][χNE1 DE; χin] * flip(rotl120_pf_honeycomb(scheme.A, dir - 1), [1 3])[DNW DE DSW]
        @tensor opt = true Tb′[χout DW; χin] := scheme.Ta[dir][χout DNE; χNE] * flip(rotl120_pf_honeycomb(scheme.B, dir - 1), 2)[DSE DW DNE] * projectors[dir][1][χNE DSE; χin]
        scheme.Ta[dir] = Ta′ / norm(Ta′)
        scheme.Tb[dir] = Tb′ / norm(Tb′)
    end
    return
end

function error_measure(Ss::Vector{T}, Ss_prev::Vector{T}) where {E, U, T <: AbstractTensorMap{E, U}}
    ϵs = [(space(S) == space(S_prev)) ? norm(S - S_prev) : Inf for (S, S_prev) in zip(Ss, Ss_prev)]
    return maximum(ϵs)
end

function lnz(scheme::CTM_honeycomb)
    return real(log(network_value(scheme)))
end

function network_value(scheme::CTM_honeycomb)
    nw_full_large = _contract_site_large(scheme)
    nw_full = _contract_site(scheme)
    nw_corners = _contract_corners(scheme)
    return (nw_full_large * nw_corners / nw_full^2)^(1 / 6)
end

function _contract_site_large(scheme::CTM_honeycomb)
    return @tensor opt = true scheme.C[1][χNW5; χNE1] *
        scheme.Ta[1][χNE1 D2; χNE2] * scheme.Tb[1][χNE2 D7; χNE3] *
        scheme.Ta[1][χNE3 D14; χNE4] * scheme.Tb[1][χNE4 D23; χNE5] *
        scheme.C[2][χNE5; χS1] *
        scheme.Ta[2][χS1 D33; χS2] * scheme.Tb[2][χS2 D32; χS3] *
        scheme.Ta[2][χS3 D31; χS4] * scheme.Tb[2][χS4 D30; χS5] *
        scheme.C[3][χS5; χNW1] *
        scheme.Ta[3][χNW1 D22; χNW2] * scheme.Tb[3][χNW2 D13; χNW3] *
        scheme.Ta[3][χNW3 D6; χNW4] * scheme.Tb[3][χNW4 D1; χNW5] *
        scheme.A[D1 D3 D4] * flip(scheme.B, [1 2])[D5 D3 D2] *
        scheme.A[D5 D7 D9] * flip(scheme.B, [1 2 3])[D12 D10 D9] *
        scheme.A[D8 D10 D11] * flip(scheme.B, [1 3])[D8 D6 D4] *
        scheme.A[D13 D15 D17] * flip(scheme.B, [1 2 3])[D18 D15 D11] *
        scheme.A[D18 D21 D25] * flip(scheme.B, [2 3])[D31 D28 D25] *
        scheme.A[D24 D28 D30] * flip(scheme.B, [1 3])[D24 D22 D17] *
        scheme.A[D12 D16 D19] * flip(scheme.B, [1 2])[D20 D16 D14] *
        scheme.A[D20 D23 D27] * flip(scheme.B, [2 3])[D33 D29 D27] *
        scheme.A[D26 D29 D32] * flip(scheme.B, [1 2 3])[D26 D21 D19]
end

function _contract_site(scheme::CTM_honeycomb)
    return @tensor opt = true scheme.C[1][χNW2; χNE1] *
        scheme.Ta[1][χNE1 D1; χNEC] * scheme.Tb[1][χNEC D2; χNE2] *
        scheme.C[2][χNE2; χS1] *
        scheme.Ta[2][χS1 D3; χSC] * scheme.Tb[2][χSC D4; χS2] *
        scheme.C[3][χS2; χNW1] *
        scheme.Ta[3][χNW1 D5; χNWC] * scheme.Tb[3][χNWC D6; χNW2] *
        scheme.A[D6 D7 D12] * flip(scheme.B, [1 2])[D8 D7 D1] *
        scheme.A[D8 D2 D9] * flip(scheme.B, [2 3])[D3 D10 D9] *
        scheme.A[D11 D10 D4] * flip(scheme.B, [1 3])[D11 D5 D12]
end

function _contract_corners(scheme::CTM_honeycomb)
    return @tensor scheme.C[1][1; 2] * scheme.C[2][2; 3] * scheme.C[3][3; 1]
end
