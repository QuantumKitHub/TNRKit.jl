"""
$(TYPEDEF)

Corner Transfer Matrix environment + Higher-Order Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T, χenv[, ctm_iter=maxiter(2.0e4), ctm_tol=trivial_convcrit(1.0e-9), ctm_obc=false, χenv_ini=2])

### Running the algorithm
    run!(::ctm_HOTRG, trunc::TruncationStrategy, criterion::stopcrit[, sweep=30, inv=false, conv_criterion=1.0e-12])

### Fields

$(TYPEDFIELDS)

### References
* [Morita & Kawashima Phys. Rev. B 103(4) (2021)](@cite morita2021)
"""
mutable struct ctm_HOTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}, TC <: AbstractTensorMap{E, S, 1, 1}, TE <: AbstractTensorMap{E, S, 2, 1}} <: TNRScheme{E, S}
    T::TT
    C2::TC
    E1::TE
    E2::TE
    χenv::Int64
    function ctm_HOTRG(
            T::TT,
            χenv::Int64;
            ctm_iter = maxiter(2.0e4),
            ctm_tol = trivial_convcrit(1.0e-9),
            ctm_obc = false,
            χenv_ini = 2,
        ) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        if E != Float64
            @error "This scheme only supports tensors with real numbers"
        end
        scheme_init = TNRKit.rCTM(T)

        @info "Finding the environment using rCTM..."
        run!(
            scheme_init,
            truncrank(χenv),
            ctm_tol & ctm_iter;
            verbosity = 0,
        )
        @info "rCTM finished"
        C2, E1, E2 = scheme_init.C2, scheme_init.E1, scheme_init.E2
        @assert BraidingStyle(sectortype(T)) == Bosonic() "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for rCTM"
        return new{E, S, TT, typeof(C2), typeof(E1)}(T, C2, E1, E2, χenv)
    end
end

function abs_tensor!(T)
    T.data .= abs.(T.data)
    return T
end

function rCTM_init_OBC(T::AbstractTensorMap{E}; χenv_ini = 2) where {E}
    Vp1 = space(T)[3]'
    Vp2 = space(T)[4]'
    S1 = sectortype(Vp1)
    V1 = Vect[S1](one(S1) => χenv_ini)
    S2 = sectortype(Vp2)
    V2 = Vect[S2](one(S2) => χenv_ini)

    C = randn(E, V1 ← V2)
    E1 = randn(E, V1 ⊗ Vp1 ← V1)
    E2 = randn(E, V2 ⊗ Vp2 ← V2)

    return abs_tensor!(C), abs_tensor!(E1), abs_tensor!(E2)
end

function corner_matrix(scheme::ctm_HOTRG)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.E1[-1 3; 1] * scheme.C2[1; 2] *
        scheme.E2[2 4; -3] * scheme.T[-2 -4; 3 4]
    return mat
end

function find_UVt(scheme::ctm_HOTRG, trunc)
    mat = corner_matrix(scheme)
    U, S, Vt = svd_trunc(mat; trunc = trunc & trunctol(atol = 1.0e-20))
    return mat, U, S, Vt
end

function vertical_move!(scheme, trunc)
    mat = corner_matrix(scheme)
    mat2 = mat * adjoint(mat)
    P1, P2 = find_P1P2(mat2, adjoint(mat2), (2, 4), (4, 2), trunc)
    @tensoropt Tnew[-1 -2; -3 -4] := P1[1 3; -1] * P2[-4; 2 4] * scheme.T[1 5; -3 2] *
        conj(scheme.T[3 5; -2 4])
    @tensoropt E2new[-1 -2; -3] := P1[1 2; -2] * scheme.E2[-1 1; 3] *
        conj(scheme.E2[-3 2; 3])

    scheme.T = Tnew
    return scheme.E2 = E2new
end

function horizontal_move!(scheme, trunc)
    mat = corner_matrix(scheme)
    mat2 = adjoint(mat) * mat
    P1, P2 = find_P1P2(mat2, adjoint(mat2), (2, 4), (4, 2), trunc)
    @tensoropt Tnew[-1 -2; -3 -4] := P1[3 4; -2] * P2[-3; 1 2] * scheme.T[5 4; 2 -4] *
        conj(scheme.T[5 3; 1 -1])
    @tensoropt E1new[-1 -2; -3] := P1[1 2; -2] * scheme.E1[3 2; -3] *
        conj(scheme.E1[3 1; -1])

    scheme.T = Tnew
    return scheme.E1 = E1new
end

function step!(
        scheme::ctm_HOTRG,
        trunc;
        sweep = 30,
        χenv = dim(scheme.C2.space.domain),
        inv = false,
    )
    vertical_move!(scheme, trunc)
    horizontal_move!(scheme, trunc)

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
        scheme::ctm_HOTRG, trunc::TruncationStrategy, criterion::stopcrit;
        sweep = 30, inv = false, conv_criterion = 1.0e-12
    )
    area = 1
    lnz = 0.0

    for i in 1:(criterion.n)
        area *= 4.0
        tr_norm = step!(scheme, trunc; sweep = sweep, inv = inv)
        lnz += log(tr_norm) / area
        if abs(log(abs(tr_norm)) / area) <= conv_criterion
            @info "CTM-HOTRG converged after $i iterations!"
            break
        end
    end
    return lnz
end

function Base.show(io::IO, scheme::ctm_HOTRG)
    println(io, "ctm_HOTRG - Corner Transfer Matrix Environment + HOTRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C2))")
    println(io, "  * E: $(summary(scheme.E1))")
    println(io, "  * E: $(summary(scheme.E2))")
    return nothing
end
