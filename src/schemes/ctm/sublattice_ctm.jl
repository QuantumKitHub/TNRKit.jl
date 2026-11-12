"""
-- CTMRG for A-B sublattice systems

┌────┐  ┌───┐  ┌───┐  ┌────┐
│Ctl1├──┤EtB├──┤EtA├──┤Ctr1│
└─┬──┘  └─┬─┘  └─┬─┘  └─┬──┘
┌─┼──┐  ┌─┼─┐  ┌─┼─┐  ┌─┼──┐
│ElB ├──┤ A ├──┤ B ├──┤ErA │
└─┬──┘  └─┬─┘  └─┬─┘  └─┬──┘
┌─┼──┐  ┌─┼─┐  ┌─┼─┐  ┌─┼──┐
│ElA ├──┤ B ├──┤ A ├──┤ErB │
└─┬──┘  └─┬─┘  └─┬─┘  └─┬──┘
┌─┼──┐  ┌─┼─┐  ┌─┼─┐  ┌─┼──┐
│Cbl1├──┤EbA├──┤EbB├──┤Cbr1│
└────┘  └───┘  └───┘  └────┘

┌────┐  ┌───┐  ┌───┐  ┌────┐
│Ctl2├──┤EtA├──┤EtB├──┤Ctr2│
└─┬──┘  └─┬─┘  └─┬─┘  └─┬──┘
┌─┼──┐  ┌─┼─┐  ┌─┼─┐  ┌─┼──┐
│ElA ├──┤ B ├──┤ A ├──┤ErB │
└─┬──┘  └─┬─┘  └─┬─┘  └─┬──┘
┌─┼──┐  ┌─┼─┐  ┌─┼─┐  ┌─┼──┐
│ElB ├──┤ A ├──┤ B ├──┤ErA │
└─┬──┘  └─┬─┘  └─┬─┘  └─┬──┘
┌─┼──┐  ┌─┼─┐  ┌─┼─┐  ┌─┼──┐
│Cbl2├──┤EbB├──┤EbA├──┤Cbr2│
└────┘  └───┘  └───┘  └────┘

"""
mutable struct Sublattice_CTM
    TA::AbstractTensorMap
    TB::AbstractTensorMap
    Ctl1::AbstractTensorMap
    Ctr1::AbstractTensorMap
    Cbr1::AbstractTensorMap
    Cbl1::AbstractTensorMap
    Ctl2::AbstractTensorMap
    Ctr2::AbstractTensorMap
    Cbr2::AbstractTensorMap
    Cbl2::AbstractTensorMap
    ElA::AbstractTensorMap
    ElB::AbstractTensorMap
    EbA::AbstractTensorMap
    EbB::AbstractTensorMap
    ErA::AbstractTensorMap
    ErB::AbstractTensorMap
    EtA::AbstractTensorMap
    EtB::AbstractTensorMap
end

Sublattice_CTM(TA, TB; bc = ones, bc_free = false) =
    Sublattice_CTM(TA, TB, CTM_init(TA, TB; bc, bc_free)...)

function lnz(ctm::Sublattice_CTM)
    A = tr(prod(contract_C1s(ctm)))
    B = tr(ρA(ctm))
    left = ctm.Cbl1 * ctm.Ctl1
    right = ctm.Ctr1 * ctm.Cbr1
    @tensor opt = true C =
        left[1; 2] *
        ctm.EtB[2 7; 3] *
        ctm.EtA[3 8; 4] *
        right[4; 5] *
        ctm.EbB[5 8; 6] *
        ctm.EbA[6 7; 1]
    top = ctm.Ctl1 * ctm.Ctr1
    bottom = ctm.Cbr1 * ctm.Cbl1
    @tensor opt = true D =
        top[1; 2] *
        ctm.ErA[2 7; 3] *
        ctm.ErB[3 8; 4] *
        bottom[4; 5] *
        ctm.ElA[5 8; 6] *
        ctm.ElB[6 7; 1]
    return log(abs(A * B / (C * D))) / 4
end

ρA(ctm::Sublattice_CTM) = ctm.Ctl1 * ctm.Ctr1 * ctm.Cbr1 * ctm.Cbl1

function CTM_init(TA, TB; bc = ones, bc_free = false)
    elt = eltype(TA)
    Vps_A = [space(TA)[i]' for i in 1:4]
    Vps_B = [space(TB)[i]' for i in 1:4]
    V = oneunit(Vps_A[1])
    if bc_free
        V = Vps_A[1]
    end
    C = bc(elt, V ← V)
    ElA, EbA, EtA, ErA = [bc(elt, V ⊗ Vps_B[i] ← V) for i in 1:4]
    ElB, EbB, EtB, ErB = [bc(elt, V ⊗ Vps_A[i] ← V) for i in 1:4]
    return C, C, C, C, C, C, C, C, ElA, ElB, EbA, EbB, ErA, ErB, EtA, EtB
end

function LinearAlgebra.normalize!(ctm::Sublattice_CTM)
    normalize!(ctm.Ctl1)
    normalize!(ctm.Ctr1)
    normalize!(ctm.Cbr1)
    normalize!(ctm.Cbl1)

    normalize!(ctm.Ctl2)
    normalize!(ctm.Ctr2)
    normalize!(ctm.Cbr2)
    normalize!(ctm.Cbl2)

    normalize!(ctm.EtA)
    normalize!(ctm.ErA)
    normalize!(ctm.EbA)
    normalize!(ctm.ElA)
    normalize!(ctm.EtB)
    normalize!(ctm.ErB)
    normalize!(ctm.EbB)
    normalize!(ctm.ElB)
    return ctm
end

function corner_spectrum(ctm::Sublattice_CTM)
    rho = ρA(ctm)
    S = LinearAlgebra.svdvals!(rho)
    return normalize!(S, 1)
end

function contract_C1s(ctm::Sublattice_CTM)
    Ctl = block_four_corner(ctm.TA, ctm.Ctl1, ctm.ElB, ctm.EtB)
    Ctr = block_four_corner(rotate_T(ctm.TB), ctm.Ctr1, ctm.EtA, ctm.ErA)
    Cbr = block_four_corner(rotate_T(ctm.TA, num = 2), ctm.Cbr1, ctm.ErB, ctm.EbB)
    Cbl = block_four_corner(rotate_T(ctm.TB, num = 3), ctm.Cbl1, ctm.EbA, ctm.ElA)
    return Ctl, Ctr, Cbr, Cbl
end

function contract_C2s(ctm::Sublattice_CTM)
    Ctl = block_four_corner(ctm.TB, ctm.Ctl2, ctm.ElA, ctm.EtA)
    Ctr = block_four_corner(rotate_T(ctm.TA), ctm.Ctr2, ctm.EtB, ctm.ErB)
    Cbr = block_four_corner(rotate_T(ctm.TB, num = 2), ctm.Cbr2, ctm.ErA, ctm.EbA)
    Cbl = block_four_corner(rotate_T(ctm.TA, num = 3), ctm.Cbl2, ctm.EbB, ctm.ElB)
    return Ctl, Ctr, Cbr, Cbl
end

function CTM_projectors(Cs, trunc)
    Ctl, Ctr, Cbr, Cbl = Cs
    ρt = Ctl * Ctr
    ρb = Cbr * Cbl
    R1, R2 = find_P1P2(ρt, ρb, (3, 4), (1, 2), trunc)
    L1, L2 = find_P1P2(ρb, ρt, (3, 4), (1, 2), trunc)
    ρr = Ctr * Cbr
    ρl = Cbl * Ctl
    T1, T2 = find_P1P2(ρl, ρr, (3, 4), (1, 2), trunc)
    B1, B2 = find_P1P2(ρr, ρl, (3, 4), (1, 2), trunc)
    Vt_list = [L1, T1, R1, B1]
    U_list = [L2, T2, R2, B2]
    return Vt_list, U_list
end

function update_corners!(ctm::Sublattice_CTM, C1s, C2s, Us)
    Vt_list1, Vt_list2, U_list1, U_list2 = Us

    Ctl_new1, Ctr_new1, Cbr_new1, Cbl_new1 = C1s
    ctm.Ctl1 = U_list1[1] * Ctl_new1 * Vt_list1[2]
    ctm.Ctr1 = U_list1[2] * Ctr_new1 * Vt_list1[3]
    ctm.Cbr1 = U_list1[3] * Cbr_new1 * Vt_list1[4]
    ctm.Cbl1 = U_list1[4] * Cbl_new1 * Vt_list1[1]

    Ctl_new2, Ctr_new2, Cbr_new2, Cbl_new2 = C2s
    ctm.Ctl2 = U_list2[1] * Ctl_new2 * Vt_list2[2]
    ctm.Ctr2 = U_list2[2] * Ctr_new2 * Vt_list2[3]
    ctm.Cbr2 = U_list2[3] * Cbr_new2 * Vt_list2[4]
    ctm.Cbl2 = U_list2[4] * Cbl_new2 * Vt_list2[1]
    return 0
end

function update_edges!(ctm::Sublattice_CTM, (Vt_list1, Vt_list2, U_list1, U_list2))
    EtA = contract_E(ctm.TA, ctm.EtB, U_list2[2], Vt_list1[2])
    EtB = contract_E(ctm.TB, ctm.EtA, U_list1[2], Vt_list2[2])
    ctm.EtA = EtA
    ctm.EtB = EtB
    ErA = contract_E(rotate_T(ctm.TA; num = 1), ctm.ErB, U_list1[3], Vt_list2[3])
    ErB = contract_E(rotate_T(ctm.TB; num = 1), ctm.ErA, U_list2[3], Vt_list1[3])
    ctm.ErA = ErA
    ctm.ErB = ErB
    EbA = contract_E(rotate_T(ctm.TA; num = 2), ctm.EbB, U_list2[4], Vt_list1[4])
    EbB = contract_E(rotate_T(ctm.TB; num = 2), ctm.EbA, U_list1[4], Vt_list2[4])
    ctm.EbA = EbA
    ctm.EbB = EbB
    ElA = contract_E(rotate_T(ctm.TA; num = 3), ctm.ElB, U_list1[1], Vt_list2[1])
    ElB = contract_E(rotate_T(ctm.TB; num = 3), ctm.ElA, U_list2[1], Vt_list1[1])
    ctm.ElA = ElA
    ctm.ElB = ElB
    return 0
end

function step!(ctm::Sublattice_CTM, trunc::TruncationStrategy)
    C1s = contract_C1s(ctm)
    C2s = contract_C2s(ctm)

    Vt_list1, U_list1 = CTM_projectors(C1s, trunc)
    Vt_list2, U_list2 = CTM_projectors(C2s, trunc)

    Us = (Vt_list1, Vt_list2, U_list1, U_list2)
    update_corners!(ctm, C1s, C2s, Us)
    update_edges!(ctm, Us)

    normalize!(ctm)
    return corner_spectrum(ctm)
end


function run!(
        ctm::Sublattice_CTM,
        trunc::TruncationStrategy,
        criterion::maxiter;
        conv_criterion = 1.0e-8,
        verbosity = 1,
    )
    ES = corner_spectrum(ctm)
    crit = true
    steps = 0
    hist = []
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting CTM calculation\n $(ctm)\n"
        while crit
            ES_new = step!(ctm, trunc)
            if size(ES) == size(ES_new)
                normdiff = norm(ES - ES_new)
                @infov 2 "Step $(steps + 1), |ES - ES_new| = $(normdiff)"
                push!(hist, normdiff)
                if normdiff < conv_criterion
                    @infov 1 "CTM converged after $(steps + 1) iterations"
                    break
                end
            end
            ES = ES_new
            steps += 1
            crit = criterion(steps, nothing)
        end
        if steps == criterion.n
            @infov 1 "CTM reached the maximum iteration $(steps)"
        end
    end
    return lnz(ctm)
end

function Base.show(io::IO, scheme::Sublattice_CTM)
    println(
        io,
        "CTMRG - Corner Transfer Matrix Renormalization Group of sublattice systems",
    )
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end
