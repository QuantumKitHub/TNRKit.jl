"""
    $(SIGNATURES)

Calculates the Ground State Degeneracy (GSD) from the fixed-point tensor of a TNRScheme,
using the eigenvalues of the transfer matrix. The GSD is the exponential of the Shannon entropy.
"""
function ground_state_degeneracy(T::AbstractTensorMap, unitcell::Int = 1)
    tm = _row_transfer_matrix(T, unitcell)
    return _ground_state_degeneracy(tm)
end
function ground_state_degeneracy(TA::AbstractTensorMap, TB::AbstractTensorMap)
    # 2-column transfer matrix
    @tensor tm[-1 -2; -3 -4] := TA[-1 1; 3 2] * TB[2 6; 4 -3] *
        TB[-2 3; 1 5] * TA[5 4; 6 -4]
    return _ground_state_degeneracy(tm)
end
ground_state_degeneracy(scheme::TNRScheme; unitcell::Int = 1) = ground_state_degeneracy(scheme.T, unitcell)
function ground_state_degeneracy(scheme::BTRG; unitcell::Int = 1)
    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    return ground_state_degeneracy(T_unit, unitcell)
end

ground_state_degeneracy(scheme::LoopTNR) = ground_state_degeneracy(scheme.TA, scheme.TB)
# helper function
function _ground_state_degeneracy(tm::AbstractTensorMap{E, S, N, N}) where {E, S, N}
    D, _ = eig_full(tm)
    D = D / tr(D)
    evs = filter(!iszero, abs.(D.data))
    entropy = -sum(evs .* log.(evs))
    return exp(entropy)
end

"""
$(SIGNATURES)
    
Calculates the Gu-Wen ratio X1 and X2 from the fixed-point tensor(s).
The Gu-Wen ratios are related to the Ground state Degeneracy and the the scaling dimensions. See references.

# References
* [Zheng-Cheng Gu & Xiao-Gang Wen. PhysRevB.80.155131](@cite gu2009)
* [Satoshi Morita et al. arxiv:2512.03395](@cite morita2025b)
"""
function gu_wen_ratio(T::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    one_norm = norm(@tensor T[1 2; 2 1])
    two_norm_X1 = norm(@tensor T[1 2; 2 3] * T[3 4; 4 1])
    two_norm_X2 = norm(@tensor T[1 2; 3 4] * T[4 3; 2 1])
    X1 = (one_norm^2) / (two_norm_X1)
    X2 = (one_norm^2) / (two_norm_X2)
    return X1, X2
end
function gu_wen_ratio(
        TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}
    ) where {E, S}
    one_norm = norm(
        @tensor opt = true TA[1 2; 3 4] * TB[4 5; 6 1] *
            TB[7 3; 2 8] * TA[8 6; 5 7]
    )
    two_norm_X1 = norm(
        @tensor opt = true TA[1 2; 3 4] * TB[4 5; 6 7] *
            TA[7 8; 9 10] * TB[10 11; 12 1] *
            TB[13 3; 2 14] * TA[14 6; 5 15] * TB[15 9; 8 16] * TA[16 12; 11 13]
    )
    two_norm_X2 = norm(
        @tensor opt = true TA[1 2; 3 4] * TB[4 5; 6 7] *
            TA[7 8; 9 10] * TB[10 11; 12 1] *
            TB[13 9; 2 14] * TA[14 12; 5 15] *
            TB[15 3; 8 16] * TA[16 6; 11 13]
    )
    X1 = (one_norm^2) / (two_norm_X1)
    X2 = (one_norm^2) / (two_norm_X2)
    return X1, X2
end
gu_wen_ratio(scheme::TNRScheme) = gu_wen_ratio(scheme.T)
function gu_wen_ratio(scheme::BTRG)
    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    return gu_wen_ratio(T_unit)
end
gu_wen_ratio(scheme::LoopTNR) = gu_wen_ratio(scheme.TA, scheme.TB)
