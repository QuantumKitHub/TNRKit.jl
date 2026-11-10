"""
$(TYPEDEF)

Corner Transfer Matrix Renormalization Group (one-site environment)

# Constructors
    $(FUNCTIONNAME)(T[, bc=ones, bc_free=false])

# Running the algorithm
    run!(::CTM, trunc::TruncationStrategy, criterion::maxiter[, conv_criterion=1.0e-8, verbosity=1])

# Fields

$(TYPEDFIELDS)

```
    ┌───┐ ┌──┐ ┌───┐
    │Ctl├─┤Et├─┤Ctr│
    └─┬─┘ └┬─┘ └─┬─┘
    ┌─┼─┐ ┌┼─┐ ┌─┼─┐
    │El ├─┤T ├─┤Er │
    └─┬─┘ └┬─┘ └─┬─┘
    ┌─┼─┐ ┌┼─┐ ┌─┼─┐
    │Cbl├─┤Eb├─┤Cbr│
    └───┘ └──┘ └───┘
```

# References
* [Baxter Physica A 106(1) (1981)](@cite baxter1981)
* [Nishino & Okunishi J. Phys. Soc. Jpn. 65 (1996)](@cite nishino1996)
"""
mutable struct CTM{E, S, TT <: AbstractTensorMap{E, S, 2, 2}, TC <: AbstractTensorMap{E, S, 1, 1}, TE <: AbstractTensorMap{E, S, 2, 1}} <: TNRScheme{E, S}
    T::TT
    Ctl::TC
    Ctr::TC
    Cbr::TC
    Cbl::TC
    El::TE
    Eb::TE
    Et::TE
    Er::TE
end

CTM(T; bc = ones, bc_free = false) = CTM(T, CTM_init(T; bc, bc_free)...)

function lnz(ctm::CTM)
    @tensor opt = true A =
        ctm.T[9 10; 11 12] *
        ctm.Ctl[1; 2] *
        ctm.Et[2 11; 3] *
        ctm.Ctr[3; 4] *
        ctm.Er[4 12; 5] *
        ctm.Cbr[5; 6] *
        ctm.Eb[6 10; 7] *
        ctm.Cbl[7; 8] *
        ctm.El[8 9; 1]

    B = tr(ρA(ctm))

    @tensor opt = true C =
        ctm.Ctl[1; 2] *
        ctm.Et[2 7; 3] *
        ctm.Ctr[3; 4] *
        ctm.Cbr[4; 5] *
        ctm.Eb[5 7; 6] *
        ctm.Cbl[6; 1]

    @tensor opt = true D =
        ctm.Ctl[1; 2] *
        ctm.Ctr[2; 3] *
        ctm.Er[3 7; 4] *
        ctm.Cbr[4; 5] *
        ctm.Cbl[5; 6] *
        ctm.El[6 7; 1]
    return log(abs(A * B / (C * D)))
end

ρA(ctm::CTM) = ctm.Ctl * ctm.Ctr * ctm.Cbr * ctm.Cbl

function CTM_init(T; bc = ones, bc_free = false)
    elt = eltype(T)
    Vps = [space(T)[i]' for i in 1:4]
    V = oneunit(Vps[1])
    if bc_free
        V = Vps[1]
    end
    C = bc(elt, V ← V)
    El, Eb, Et, Er = [bc(elt, V ⊗ Vps[i] ← V) for i in 1:4]
    return C, C, C, C, El, Eb, Et, Er
end

function LinearAlgebra.normalize!(ctm::CTM)
    normalize!(ctm.Ctl)
    normalize!(ctm.Ctr)
    normalize!(ctm.Cbr)
    normalize!(ctm.Cbl)
    normalize!(ctm.Et)
    normalize!(ctm.Er)
    normalize!(ctm.Eb)
    normalize!(ctm.El)
    return ctm
end

"""
┌──┐┌──┐   
│C ┼┼E2┼─ -3
└┬─┘└┬─┘   
┌┼─┐┌┼─┐   
│E1┼┤T ┼─ -4
└┬─┘└┬─┘   
 │   │     
-1   -2    
"""

function block_four_corner(T, C, E1, E2)
    @tensor opt = true Cnew[-1 -2; -3 -4] :=
        T[3 -2; 4 -4] * C[1; 2] * E1[-1 3; 1] * E2[2 4; -3]
    return Cnew
end

# Rotate the tensor T by 90 degrees counter-clockwise
function rotate_T(T; num = 1)
    Tnew = copy(T)
    for _ in 1:num
        Tnew = permute(Tnew, ((3, 1), (4, 2)))
    end
    return Tnew
end

function contract_E(T, E, U, Vt)
    @tensor opt = true Enew[-1 -2; -3] := T[2 -2; 3 5] * E[1 3; 4] * U[-1; 1 2] * Vt[4 5; -3]
    return Enew
end

function corner_spectrum(ctm::CTM)
    rho = ρA(ctm)
    S = LinearAlgebra.svdvals!(rho)
    return normalize!(S, 1)
end

function step!(ctm::CTM, trunc::TruncationStrategy)
    Ctl_new = block_four_corner(ctm.T, ctm.Ctl, ctm.El, ctm.Et)
    Ctr_new = block_four_corner(rotate_T(ctm.T), ctm.Ctr, ctm.Et, ctm.Er)
    Cbr_new = block_four_corner(rotate_T(ctm.T, num = 2), ctm.Cbr, ctm.Er, ctm.Eb)
    Cbl_new = block_four_corner(rotate_T(ctm.T, num = 3), ctm.Cbl, ctm.Eb, ctm.El)

    ρt = Ctl_new * Ctr_new
    ρb = Cbr_new * Cbl_new
    R1, R2 = find_P1P2(ρt, ρb, (3, 4), (1, 2), trunc)
    L1, L2 = find_P1P2(ρb, ρt, (3, 4), (1, 2), trunc)
    ρr = Ctr_new * Cbr_new
    ρl = Cbl_new * Ctl_new
    T1, T2 = find_P1P2(ρl, ρr, (3, 4), (1, 2), trunc)
    B1, B2 = find_P1P2(ρr, ρl, (3, 4), (1, 2), trunc)

    Vt_list = [L1, T1, R1, B1]
    U_list = [L2, T2, R2, B2]

    ctm.Ctl = U_list[1] * Ctl_new * Vt_list[2]
    ctm.Ctr = U_list[2] * Ctr_new * Vt_list[3]
    ctm.Cbr = U_list[3] * Cbr_new * Vt_list[4]
    ctm.Cbl = U_list[4] * Cbl_new * Vt_list[1]
    ctm.Et = contract_E(ctm.T, ctm.Et, U_list[2], Vt_list[2])
    ctm.Er = contract_E(rotate_T(ctm.T; num = 1), ctm.Er, U_list[3], Vt_list[3])
    ctm.Eb = contract_E(rotate_T(ctm.T; num = 2), ctm.Eb, U_list[4], Vt_list[4])
    ctm.El = contract_E(rotate_T(ctm.T; num = 3), ctm.El, U_list[1], Vt_list[1])
    normalize!(ctm)
    return corner_spectrum(ctm)
end


function run!(ctm::CTM, trunc::TruncationStrategy, criterion::maxiter; conv_criterion = 1.0e-8, verbosity = 1)
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

function Base.show(io::IO, scheme::CTM)
    println(io, "CTMRG - Corner Transfer Matrix Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
