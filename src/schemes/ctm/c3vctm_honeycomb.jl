# Based on
# https://arxiv.org/abs/2209.03428

function run!(
        scheme::c3vCTM_honeycomb, trunc::TruncationStrategy, criterion::stopcrit;
        verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), ε = $(ε)"

            ε = step!(scheme, trunc)
            steps += 1
            crit = criterion(steps, ε)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function step!(scheme::c3vCTM_honeycomb, trunc)
    D, W = calculate_projectors(scheme, trunc)

    renormalize_corners!(scheme, W)
    scheme.C /= norm(scheme.C)

    renormalize_edges!(scheme, W)
    scheme.L /= norm(scheme.R)
    scheme.R /= norm(scheme.R)
    return error_measure(scheme)
end

function calculate_projectors(scheme, trunc)
    @tensor opt = true mat[χout Dout; χin Din] := scheme.L[χout DNW; χN] * scheme.C[χN χNE] * scheme.R[χNE DE; χin] *
        scheme.T[DNW DE DC] * conj(flip(scheme.T, 1)[Din Dout DC])

    D, W = eig_trunc(mat; trunc)
    return D, W
end

function renormalize_corners!(scheme, W)
    return @tensor opt = true scheme.C[χout; χin] := scheme.L[χWc DNW; χN] * scheme.C[χN χNE] * scheme.R[χNE DE; χE] *
        scheme.T[DNW DE DC] * conj(flip(scheme.T, 1)[Din Dout DC]) *
        W[χE Din; χin] * conj(W[χWc Dout; χout])
end

function renormalize_edges!(scheme, W)
    @tensor opt = true scheme.L[χout Dout; χin] := scheme.L[χ1 D1; χ2] * scheme.T[D1 D2 D3] * conj(scheme.T[Dout D4 D3]) *
        W[χ2 D2; χin] * conj(W[χ1 D4; χout])
    return @tensor opt = true scheme.R[χout Dout; χin] := flip(scheme.R, 2)[χ1 D1; χ2] * conj(scheme.T[D2 D3 D1]) * flip(scheme.T, 3)[D2 D4 Dout] *
        W[χ2 D4; χin] * conj(W[χ1 D3; χout])
end

function error_measure(scheme)
    @tensor LHS[χout Dout; χin] := scheme.C[χout; χ] * scheme.L[χ Dout; χin]
    @tensor RHS[χout Dout; χin] := scheme.R[χout Dout; χ] * scheme.C[χ; χin]
    return norm(LHS - RHS)
end

function lnz(scheme::c3vCTM_honeycomb)
    return real(log(network_value(scheme)))
end

function network_value(scheme::c3vCTM_honeycomb)
    nw_corners = _contract_corners(scheme)
    nw_full = _contract_site_large(scheme)
    nw_L = _contract_edges_L(scheme)
    nw_R = _contract_edges_R(scheme)
    return sqrt(nw_full^3 * nw_corners / ((nw_L * nw_R)^2))
end

function _contract_corners(scheme::c3vCTM_honeycomb)
    return @tensor scheme.C[1; 2] * scheme.C[2; 3] * scheme.C[3; 4] * scheme.C[4; 5] * scheme.C[5; 6] * scheme.C[6; 1]
end

function _contract_site_large(scheme::c3vCTM_honeycomb)
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.L[χNL DNW; χNR] *
        scheme.C[χNR χNEL] * scheme.R[χNEL DE; χNER] *
        scheme.C[χNER; χSE] * scheme.C[χSE; χSL] * scheme.L[χSL DSE; χSR] *
        scheme.C[χSR; χSWL] * scheme.R[χSWL DW; χSWR] *
        scheme.T[DNW DE DC] * conj(flip(scheme.T, [1 2])[DSE DW DC])
end

function _contract_edges_L(scheme::c3vCTM_honeycomb)
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.L[χNL DNW; χNR] *
        scheme.C[χNR χNE] * scheme.C[χNE; χSEL] * scheme.L[χSEL DE; χSER] *
        scheme.C[χSER; χS] * scheme.C[χS; χSWL] * scheme.L[χSWL DSW; χSWR] *
        scheme.T[DNW DE DSW]
end

function _contract_edges_R(scheme::c3vCTM_honeycomb)
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.R[χNL DNW; χNR] *
        scheme.C[χNR χNE] * scheme.C[χNE; χSEL] * scheme.R[χSEL DE; χSER] *
        scheme.C[χSER; χS] * scheme.C[χS; χSWL] * scheme.R[χSWL DSW; χSWR] *
        conj(flip(scheme.T, [1 2 3])[DE DNW DSW])

end
