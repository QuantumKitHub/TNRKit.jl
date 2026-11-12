function ind_pair(T::AbstractTensorMap, p::Tuple)
    p2 = filter(x -> !in(x, p), allind(T))
    return p, p2
end

# QR decomposition
function R1R2(A1, A2, p1, p2; check_space = true)
    p, q1 = ind_pair(A1, p1)
    _, RA1 = left_orth(permute(A1, (q1, p1)))
    p, q2 = ind_pair(A2, p2)
    RA2, _ = right_orth(permute(A2, (p2, q2)))
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

# Find the pair of oblique projectors acting on the indices p1 of A1 and p2 of A2
#=
   ┌──┐        ┌──┐   
   │  ├◄──  ─◄─┤  │   
─◄─┤P1│        │P2├◄──
   │  ├◄──  ─◄─┤  │   
   └──┘        └──┘   
=#

function find_P1P2(A1, A2, p1, p2, trunc; check_space = true)
    R1, R2 = R1R2(A1, A2, p1, p2; check_space = check_space)
    return oblique_projector(R1, R2, trunc)
end

function oblique_projector(R1, R2, trunc; cutoff = 1.0e-16)
    mat = R1 * R2
    U, S, Vt = svd_trunc(mat; trunc = trunc & trunctol(atol = cutoff))

    P1 = R2 * adjoint(Vt) / sqrt(S)
    P2 = adjoint(U) * R1
    P2 = adjoint(adjoint(P2) / adjoint(sqrt(S)))
    return P1, P2
end

function tr_tensor(T; inv = false)
    if inv
        @tensoropt tr4 = T[1 2; 3 4] * conj(T[5 2; 3 6]) * conj(T[1 7; 8 4]) * T[5 7; 8 6]
        return (abs(tr4))^(1 / 4)
    else
        return @tensor T[1 2; 2 1]
    end
end

function rctm_step!(scheme; trunc = truncrank(dim(space(scheme.C2, 1))))
    mat, U, S, Vt = find_UVt(scheme, trunc)
    scheme.C2 = adjoint(U) * mat * adjoint(Vt)
    @tensor opt = true scheme.E1[-1 -2; -3] := scheme.E1[1 5; 3] * scheme.T[2 -2; 5 4] *
        U[3 4; -3] * conj(U[1 2; -1])
    @tensor opt = true scheme.E2[-1 -2; -3] := scheme.E2[1 5; 3] * scheme.T[-2 4; 2 5] *
        conj(Vt[-3; 3 4]) * Vt[-1; 1 2]
    normalize!(scheme.C2)
    normalize!(scheme.E1)
    normalize!(scheme.E2)
    return S
end
