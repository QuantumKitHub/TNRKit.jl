"""
$(TYPEDEF)

Corner Transfer Matrix environment + Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T, χenv[, ctm_iter=2.0e4, ctm_tol=1.0e-9])

### Running the algorithm
    run!(::ctm_TRG, trunc::TruncationStrategy, criterion::maxiter[, sweep=30, enlarge=true, inv=false, conv_criterion=1.0e-12, modified=true])

### Fields

$(TYPEDFIELDS)

### References
* [Morita & Kawashima Phys. Rev. B 103(4) (2021)](@cite morita2021)
"""
mutable struct ctm_TRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}, TC <: AbstractTensorMap{E, S, 1, 1}, TE <: AbstractTensorMap{E, S, 2, 1}} <: TNRScheme{E, S}
    T::TT
    C2::TC
    E1::TE
    E2::TE
    χenv::Int64
    function ctm_TRG(
            T::TT,
            χenv::Int64;
            ctm_iter = 2.0e4,
            ctm_tol = 1.0e-9,
        ) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        if eltype(T) != Float64
            @error "This scheme only support tensors with real numbers"
        end
        scheme_init = TNRKit.rCTM(T)
        @info "Finding the environment using rCTM..."
        TNRKit.run!(
            scheme_init,
            truncrank(χenv),
            trivial_convcrit(ctm_tol) & maxiter(ctm_iter);
            verbosity = 0,
        )
        @info "rCTM finished"
        C2, E1, E2 = scheme_init.C2, scheme_init.E1, scheme_init.E2
        @assert BraidingStyle(sectortype(T)) == Bosonic() "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for rCTM"
        return new{E, S, TT, typeof(C2), typeof(E1)}(T, C2, E1, E2, χenv)
    end
end

function corner_matrix(scheme::ctm_TRG)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.E1[-1 3; 1] * scheme.C2[1; 2] *
        scheme.E2[2 4; -3] * scheme.T[-2 -4; 3 4]
    return mat
end

function find_UVt(scheme::ctm_TRG, trunc)
    mat = corner_matrix(scheme)
    U, S, Vt = svd_trunc(mat; trunc = trunc & trunctol(atol = 1.0e-20))
    return mat, U, S, Vt
end

function Levin_decomposition(T, trunc)
    U, S, Vt = svd_trunc(permute(T, ((1, 3), (2, 4))); trunc = trunc)

    S1 = U * sqrt(S)
    S2 = sqrt(S) * Vt
    return S1, S2
end

function insert_PtoS(scheme, trunc; enlarge = true)
    S1, S2 = Levin_decomposition(scheme.T, notrunc())

    @tensoropt env_left_top[-1 -2; -3] := S1[3 4; -3] *
        scheme.E1[2 4; -1] *
        conj(scheme.E2[1 3; 5]) *
        conj(scheme.C2[2; 1]) *
        scheme.C2[-2; 5]
    @tensoropt env_right_bottom[-1; -2 -3] := S2[-1; 1 2] *
        scheme.E2[3 2; 4] *
        conj(scheme.E1[-3 1; 5]) *
        scheme.C2[-2; 3] *
        conj(scheme.C2[5; 4])
    if enlarge
        P1, P2 = find_P1P2(
            env_left_top, env_right_bottom, (3,), (1,),
            truncrank(trunc.howmany * 2)
        )
    else
        P1, P2 = find_P1P2(env_left_top, env_right_bottom, (3,), (1,), trunc)
    end
    return S1 * P1, P2 * S2
end

# find the projector for bundling two bonds in the vertical and horizontal directions
# I checked the modified version provides the same accuracy for free energy. (and cheaper)
# I pursue this because this is compatible with entanglement filtering
function hotrg_projector(scheme, trunc; modified = true)
    if modified
        return hotrg_projector_modified(scheme, trunc)
    else
        mat = corner_matrix(scheme)

        matv2 = mat * adjoint(mat)
        Pv1, Pv2 = find_P1P2(matv2, adjoint(matv2), (2, 4), (4, 2), trunc)
        math2 = adjoint(mat) * mat
        Ph1, Ph2 = find_P1P2(math2, adjoint(math2), (2, 4), (4, 2), trunc)
        return Pv1, Pv2, Ph1, Ph2
    end
end

function hotrg_projector_modified(scheme, trunc)
    @tensor mat[-1; -2 -3] := scheme.E1[-1; -2 1] * scheme.C2[1; -3]
    math2 = adjoint(mat) * mat
    Ph1, Ph2 = find_P1P2(math2, adjoint(math2), (1, 3), (3, 1), trunc)

    @tensor mat[-1 -2; -3] := scheme.C2[-1; 1] * scheme.E2[1 -2; -3]
    matv2 = mat * adjoint(mat)
    Pv1, Pv2 = find_P1P2(matv2, adjoint(matv2), (2, 4), (4, 2), trunc)
    return Pv1, Pv2, Ph1, Ph2
end

function step!(
        scheme::ctm_TRG,
        trunc::TruncationStrategy, ;
        sweep = 30,
        enlarge = true,
        inv = false,
        modified = true,
    )
    χenv = dim(scheme.C2.space.domain)
    S1, S2 = insert_PtoS(scheme, trunc; enlarge = enlarge)
    Pv1, Pv2, Ph1, Ph2 = hotrg_projector(scheme, trunc; modified)
    # My apologies for the unreadable contraction ~ A.U.
    # It just combines tensors and projectors to build the new tensor
    @tensoropt Tnew[-1 -2; -3 -4] := Pv1[2 3; -1] *
        S2[13; 1 3] *
        conj(S2[16; 1 2]) *
        Pv2[-4; 11 12] *
        S2[15; 10 11] *
        conj(S2[14 10 12]) *
        Ph1[5 6; -2] *
        S1[4 5; 13] *
        conj(S1[4 6; 14]) *
        Ph2[-3; 8 9] *
        S1[7 9; 15] *
        conj(S1[7 8; 16])
    @tensoropt E1new[-1 -2; -3] := Ph1[1 2; -2] * scheme.E1[3 2; -3] *
        conj(scheme.E1[3 1; -1])
    @tensoropt E2new[-1 -2; -3] := Pv1[1 2; -2] * scheme.E2[-1 1; 3] *
        conj(scheme.E2[-3 2; 3])
    scheme.T = Tnew
    scheme.E1 = E1new
    scheme.E2 = E2new

    tr_norm = tr_tensor(scheme.T; inv = inv)
    scheme.T /= tr_norm
    normalize!(scheme.E1)
    normalize!(scheme.E2)
    for _ in 0:sweep
        rctm_step!(scheme)
    end
    return tr_norm
end

function run!(
        scheme::ctm_TRG,
        trunc::TruncationStrategy,
        criterion::maxiter;
        sweep = 30,
        enlarge = true,
        inv = false,
        conv_criterion = 1.0e-12,
        modified = true,
    )
    area = 1
    lnz = 0.0

    steps = 0
    crit = true
    while crit
        area *= 4.0
        tr_norm = step!(scheme, trunc; sweep = sweep, enlarge = enlarge, inv = inv, modified)
        lnz += log(tr_norm) / area
        if abs(log(abs(tr_norm)) / area) <= conv_criterion
            @info "CTM-TRG converged after $steps iterations"
            break
        end
        steps += 1
        crit = criterion(steps, nothing)
    end
    return lnz
end

function Base.show(io::IO, scheme::ctm_TRG)
    println(io, "ctm_TRG - Corner Transfer Matrix Environment + TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C2))")
    println(io, "  * E: $(summary(scheme.E1))")
    println(io, "  * E: $(summary(scheme.E2))")
    return nothing
end
