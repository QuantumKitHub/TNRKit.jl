function cdl_tensor(χ::Int; χcdl = 2)
    A = randn(ℂ^χ ⊗ ℂ^χ ← ℂ^χ ⊗ ℂ^χ)
    C = randn(ℂ^χcdl ← ℂ^χcdl)

    # fusing isometory
    U = isometry(ℂ^χcdl ⊗ ℂ^χ ⊗ ℂ^χcdl, ℂ^(2 * χcdl + χ))
    U = flip(U, (2, 3))
    @tensoropt Anew[-1 -2; -3 -4] := A[2 5; 8 11] * C[3; 4] * C[7; 1] * C[6; 12] *
        C[10; 9] * U[1 2 3; -1] * U[4 5 6; -2] *
        conj(U[7 8 9; -3]) * conj(U[10 11 12; -4])
    # random rotation
    U, _, Vt = svd_compact(randn(ℂ^(2 * χcdl + χ) ← ℂ^(2 * χcdl + χ)))
    @tensoropt Anew[-1 -2; -3 -4] := Anew[1 2; 3 4] * U[1; -1] * conj(U[4; -4]) *
        Vt[2; -2] * conj(Vt[3; -3])
    return Anew
end
