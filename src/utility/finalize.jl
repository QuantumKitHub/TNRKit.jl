const simple_scheme = Union{TRG, ATRG, HOTRG}

# 1x1 unitcell finalize
function finalize!(scheme::simple_scheme)
    n = norm(@tensor scheme.T[1 2; 2 1])
    scheme.T /= n
    return n
end

function finalize!(scheme::BTRG)
    n = norm(@tensor scheme.T[1 2; 4 3] * scheme.S1[4; 2] * scheme.S2[3; 1])
    scheme.T /= n
    return n
end

# 2x2 unitcell finalize
function finalize_two_by_two!(scheme::simple_scheme)
    n = norm(
        @tensor scheme.T[7 1; 5 4] * scheme.T[4 2; 6 7] * scheme.T[3 6; 2 8] *
            scheme.T[8 5; 1 3]
    )

    scheme.T /= (n^(1 / 4))
    return n^(1 / 4)
end

function finalize_two_by_two!(scheme::BTRG)
    n′ = @tensor begin
        scheme.T[11 1; 9 8] *
            scheme.S2[8; 2] *
            scheme.T[2 6; 10 11] *
            scheme.S1[3; 6] *
            scheme.T[7 10; 3 12] *
            scheme.S2[4; 7] *
            scheme.T[12 9; 5 4] *
            scheme.S1[5; 1]
    end
    n = norm(n′)
    scheme.T /= (n^(1 / 4))
    return n^(1 / 4)
end

function finalize!(scheme::LoopTNR)
    T1 = permute(scheme.TA, ((1, 2), (4, 3)))
    T2 = permute(scheme.TB, ((1, 2), (4, 3)))
    n = norm(
        @tensor opt = true T1[1 2; 3 4] * T2[3 5; 1 6] *
            T2[7 4; 8 2] * T1[8 6; 7 5]
    )

    scheme.TA /= n^(1 / 4)
    scheme.TB /= n^(1 / 4)
    return n^(1 / 4)
end

function finalize!(scheme::ATRG_3D)
    n = norm(@tensor scheme.T[1 1; 2 3 2 3])
    scheme.T /= n
    return n
end

function finalize!(scheme::HOTRG_3D)
    n = norm(@tensor scheme.T[1 1; 2 3 2 3])
    scheme.T /= n
    return n
end

function finalize!(scheme::SLoopTNR)
    tr_norm = trnorm_2x2(scheme.T)
    scheme.T /= tr_norm^0.25
    return tr_norm^0.25
end

# finalize! for ImpurityTRG
function finalize!(scheme::ImpurityTRG)
    # First normalize everything by the pure tensor
    npure = norm(@tensor scheme.T[1 2; 2 1])
    scheme.T_imp1 /= npure
    scheme.T_imp2 /= npure
    scheme.T_imp3 /= npure
    scheme.T_imp4 /= npure
    scheme.T /= npure

    # Then calculate the contracted/traced 4 impurity tensors
    nimp = norm(@tensoropt scheme.T_imp1[5 4; 6 1] * scheme.T_imp2[1 2; 7 5] * scheme.T_imp3[3 7; 2 8] * scheme.T_imp4[8 6; 4 3])

    return npure, nimp
end

# finalize! for ImpurityHOTRG
function finalize!(scheme::ImpurityHOTRG)
    n = norm(@tensor scheme.T[1 2; 2 1])
    n_11 = norm(@tensor scheme.T_imp_order1_1[1 2; 2 1])
    n_12 = norm(@tensor scheme.T_imp_order1_2[1 2; 2 1])
    n_2 = norm(@tensor scheme.T_imp_order2[1 2; 2 1])
    scheme.T /= n
    scheme.T_imp_order1_1 /= n
    scheme.T_imp_order1_2 /= n
    scheme.T_imp_order2 /= n
    return n, n_11, n_12, n_2
end

# Finalizers for CorrelationHOTRG
function finalize_phase1!(scheme::CorrelationHOTRG)
    n = norm(@tensor scheme.Tpure[1 2; 2 1])
    n_1 = norm(@tensor scheme.Timp1[1 2; 2 1])
    n_2 = norm(@tensor scheme.Timp2[1 2; 2 1])

    scheme.Tpure /= n
    scheme.Timp1 /= n
    scheme.Timp2 /= n

    return n, n_1, n_2
end

function finalize_phase23!(scheme::CorrelationHOTRG)
    n = norm(@tensor scheme.Tpure[1 2; 2 1])
    n_imp = norm(@tensor scheme.Timp_final[1 2; 2 1])

    scheme.Tpure /= n
    scheme.Timp_final /= n

    return n, n_imp, n_imp
end

# cft data finalize
function finalize_cftdata!(scheme::TNRScheme)
    finalize!(scheme)
    return CFTData(scheme)
end

CFT_Finalizer = Finalizer(finalize_cftdata!, CFTData)

function finalize_cftdata!(scheme::SLoopTNR) # TODO: remove this
    tr_norm = trnorm_2x2(scheme.T)
    scheme.T /= tr_norm^0.25
    Tflip = flip(scheme.T, (1, 2, 3, 4))
    @tensoropt mat[-1 -2; -3 -4] := scheme.T[1 3; -1 2] * Tflip[1 4; -2 2] *
        Tflip[5 3; -3 6] * scheme.T[5 4; -4 6]
    val, vec = eig_full(mat)
    val = sort(real(val).data; rev = true)
    data = -log.(abs.(val ./ val[1])) / 2 / π
    return data
end

# TODO: add Finalizers for CFT and central charge
two_by_two_Finalizer = Finalizer(finalize_two_by_two!, Float64)

# Finalizer for ground state degeneracy
function finalize_groundstatedegeneracy!(scheme::TNRScheme)
    finalize!(scheme)
    return ground_state_degeneracy(scheme; unitcell = 1)
end

function finalize_groundstatedegeneracy!(scheme::BTRG)
    finalize!(scheme)
    return ground_state_degeneracy(scheme; unitcell = 1)
end

function finalize_groundstatedegeneracy!(scheme::LoopTNR)
    finalize!(scheme)
    return ground_state_degeneracy(scheme; unitcell = 2)
end

GSDegeneracy_Finalizer = Finalizer(finalize_groundstatedegeneracy!, Float64)

#Finalizer for Gu-Wen ratio
function finalize_gu_wen_ratio!(scheme::TNRScheme)
    finalize!(scheme)
    return gu_wen_ratio(scheme)
end

function finalize_gu_wen_ratio!(scheme::BTRG)
    finalize!(scheme)
    return gu_wen_ratio(scheme)
end

function finalize_gu_wen_ratio!(scheme::LoopTNR)
    finalize!(scheme)
    return gu_wen_ratio(scheme)
end

const guwenratio_Finalizer = Finalizer(finalize_gu_wen_ratio!, Tuple{Float64, Float64})
