"""
    $(SIGNATURES)

Calculates the Ground State Degeneracy (GSD) from the fixed-point tensor of a TNRScheme,
using the eigenvalues of the transfer matrix. The GSD is the exponential of the Shannon entropy.
"""
function ground_state_degeneracy(scheme::TNRScheme{E}, unitcell::Int = 1) where {E}
    # Construct contraction indices
    indices = Vector{NTuple{4, Int}}(undef, unitcell)
    for i in 1:unitcell
        indices[i] = (i, -i, -(i + unitcell), i + 1)
    end
    indices[end] = (unitcell, -unitcell, -(unitcell + unitcell), 1)

    # Contract tensors
    Ts = fill(scheme.T, unitcell)
    T = ncon(Ts, indices)

    # Construct static tuple indices
    outinds = ntuple(i -> i, unitcell)
    ininds = ntuple(i -> unitcell + i, unitcell)

    T = permute(T, (outinds, ininds))

    # Compute normalized eigenvalues
    D, _ = eig_full(T)
    D = D / tr(D)
    vals = filter(!iszero, abs.(D.data))
    # Shannon entropy (stable + efficient)
    S = 0.0
    for v in vals
        ev = abs(v)
        if ev > 0
            S -= ev * log(ev)
        end
    end

    return exp(S)
end
function ground_state_degeneracy(scheme::BTRG{E}; unitcell::Int = 1) where {E}
    indices = Vector{NTuple{4, Int}}(undef, unitcell)
    for i in 1:unitcell
        indices[i] = (i, -i, -(i + unitcell), i + 1)
    end
    indices[end] = (unitcell, -unitcell, -(unitcell + unitcell), 1)

    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    T = ncon(fill(T_unit, unitcell), indices)

    # Construct static tuple indices
    outinds = ntuple(i -> i, unitcell)
    ininds = ntuple(i -> unitcell + i, unitcell)

    T = permute(T, (outinds, ininds))
    D, _ = eig_full(T)
    D = D / tr(D)
    vals = filter(!iszero, abs.(D.data))
    # Shannon entropy (stable + efficient)
    S = 0.0
    for v in vals
        ev = abs(v)
        if ev > 0
            S -= ev * log(ev)
        end
    end

    return exp(S)
end
function ground_state_degeneracy(scheme::LoopTNR{E}) where {E}
    norm_const = area_term(scheme.TA, scheme.TB)
    T1 = scheme.TA / abs(norm_const)^(1 / 4)
    T2 = scheme.TB / abs(norm_const)^(1 / 4)

    @tensor T_unit[-1 -2; -3 -4] := T1[-1 1; 3 2] * T2[2 6; 4 -3] *
        T2[-2 3; 1 5] * T1[5 4; 6 -4]

    D, _ = eig_full(T_unit)
    D = D / tr(D)
    vals = filter(!iszero, abs.(D.data))
    # Shannon entropy (stable + efficient)
    S = 0.0
    for v in vals
        ev = abs(v)
        if ev > 0
            S -= ev * log(ev)
        end
    end

    return exp(S)
end

"""
$(SIGNATURES)
    
Calculates the Gu-Wen ratio X1 and X2 from the fixed-point tensor of a TNRScheme.
The Gu-Wen ratios are related to the Ground state Degeneracy and the the scaling dimensions. See references.

### References
* [Zheng-Cheng Gu & Xiao-Gang Wen. PhysRevB.80.155131](@cite gu2009)
* [Satoshi Morita et al. arxiv:2512.03395](@cite morita2025)
"""
function gu_wen_ratio(scheme::TNRScheme{E}) where {E}
    T_unit = scheme.T

    one_norm = norm(@tensor T_unit[1 2; 2 1])
    two_norm_X1 = norm(@tensor T_unit[1 2; 2 3] * T_unit[3 4; 4 1])
    two_norm_X2 = norm(@tensor T_unit[1 2; 3 4] * T_unit[4 3; 2 1])

    X1 = (one_norm^2) / (two_norm_X1)
    X2 = (one_norm^2) / (two_norm_X2)
    return X1, X2
end
function gu_wen_ratio(scheme::BTRG{E}) where {E}
    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]

    one_norm = norm(@tensor T_unit[1 2; 2 1])
    two_norm_X1 = norm(@tensor T_unit[1 2; 2 3] * T_unit[3 4; 4 1])
    two_norm_X2 = norm(@tensor T_unit[1 2; 3 4] * T_unit[4 3; 2 1])

    X1 = (one_norm^2) / (two_norm_X1)
    X2 = (one_norm^2) / (two_norm_X2)
    return X1, X2
end
function gu_wen_ratio(scheme::LoopTNR{E}) where {E}
    T1 = scheme.TA
    T2 = scheme.TB
    one_norm = norm(
        @tensor opt = true T1[1 2; 3 4] * T2[4 5; 6 1] *
            T2[7 3; 2 8] * T1[8 6; 5 7]
    )

    two_norm_X1 = norm(
        @tensor opt = true T1[1 2; 3 4] * T2[4 5; 6 7] *
            T1[7 8; 9 10] * T2[10 11; 12 1] *
            T2[13 3; 2 14] * T1[14 6; 5 15] * T2[15 9; 8 16] * T1[16 12; 11 13]
    )

    two_norm_X2 = norm(
        @tensor opt = true T1[1 2; 3 4] * T2[4 5; 6 7] *
            T1[7 8; 9 10] * T2[10 11; 12 1] *
            T2[13 9; 2 14] * T1[14 12; 5 15] *
            T2[15 3; 8 16] * T1[16 6; 11 13]
    )

    X1 = (one_norm^2) / (two_norm_X1)
    X2 = (one_norm^2) / (two_norm_X2)
    return X1, X2
end
