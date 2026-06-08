function P_tensor()
    P = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)
    for (pi1, pj1, pi2, pj2, i1, j1, i2, j2) in Iterators.product([0:1 for _ in 1:8]...)
        P[pi1 + 1, pj1 + 1, pi2 + 1, pj2 + 1, i1 + 1, j1 + 1, i2 + 1, j2 + 1] =
            i1 * (j1 + j2 + pi1 + pi2) + i2 * (j2 + pi1 + pi2) + pj1 * (pi1 + pi2) + pj2 * pi2 + pi1 + pi2
    end
    return P
end

function gross_neveu_8_leg_tensor(μ::Number, m::Number, g::Number; T::Type{<:Complex} = ComplexF64)
    # Manually defining the A and A_bar tensors
    A = zeros(T, 2, 2, 2, 2)
    A[2, 2, 1, 1] = 1 + 1im
    A[1, 1, 2, 2] = -1 - 1im
    A[2, 1, 1, 2] = 1 - 1im
    A[2, 1, 2, 1] = 2
    A[1, 2, 1, 2] = -2im
    A[1, 2, 2, 1] = 1 - 1im

    A_bar = zeros(T, 2, 2, 2, 2)
    A_bar[2, 2, 1, 1] = -1 + 1im
    A_bar[1, 1, 2, 2] = 1 - 1im
    A_bar[2, 1, 1, 2] = -1 - 1im
    A_bar[2, 1, 2, 1] = -2
    A_bar[1, 2, 1, 2] = -2im
    A_bar[1, 2, 2, 1] = -1 - 1im

    # Utility Kronecker delta function
    δ(x, y) = ==(x, y)

    t = zeros(T, 2, 2, 2, 2, 2, 2, 2, 2)
    P = P_tensor()
    V = Vect[FermionParity](0 => 1, 1 => 1)
    for (pi1, pj1, pi2, pj2, i1, j1, i2, j2) in Iterators.product([0:1 for _ in 1:8]...)
        p = P[pi1 + 1, pj1 + 1, pi2 + 1, pj2 + 1, i1 + 1, j1 + 1, i2 + 1, j2 + 1]
        t[pi1 + 1, pj1 + 1, pi2 + 1, pj2 + 1, i2 + 1, j2 + 1, i1 + 1, j1 + 1] =
            ((-1)^p) * exp(0.5 * μ * (i2 - j2 + pi2 - pj2)) * ((1 / sqrt(2))^(i1 + i2 + j1 + j2 + pi1 + pi2 + pj1 + pj2)) *
            (
            ((m + 2)^2 + 2 * g^2) * δ(i1 + i2 + pj1 + pj2, 0) * δ(j1 + j2 + pi1 + pi2, 0) -
                (m + 2) * δ(i1 + i2 + pj1 + pj2, 1) * δ(j1 + j2 + pi1 + pi2, 1) -
                ((-1)^(i1 + i2 + j2 + pi1)) * (1im^(i2 + j2 + pi2 + pj2)) * (m + 2) * δ(i1 + i2 + pj1 + pj2, 1) * δ(j1 + j2 + pi1 + pi2, 1) -
                A_bar[i1 + 1, i2 + 1, pj1 + 1, pj2 + 1] * A[j1 + 1, j2 + 1, pi1 + 1, pi2 + 1]
        )

    end
    return TensorMap(t, V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V)
end


"""
    gross_neveu_start([::Type{FermionParity}], μ::Number, m::Number, g::Number; T::Type{<:Complex} = ComplexF64)

Constructs the partition function tensor for the Gross-Neveu model with given parameters `μ`, `m`, and `g`.
Compatible with explicit fermion parity symmetry on each of its spaces.

# References
* [Akiyama et. al. J. Phys.: Condens. Matter 36 (2024) 343002](@cite akiyama2024)
"""
function gross_neveu_start(μ::Number, m::Number, g::Number; kwargs...)
    return gross_neveu_start(FermionParity, μ, m, g; kwargs...)
end
function gross_neveu_start(::Type{FermionParity}, μ::Number, m::Number, g::Number; T::Type{<:Complex} = ComplexF64)
    T_unfused = gross_neveu_8_leg_tensor(μ, m, g; T = T)
    V = Vect[FermionParity](0 => 1, 1 => 1)
    U = isometry(fuse(V, V), V ⊗ V)
    Udg = adjoint(U)

    @tensor T_fused[-1 -2; -3 -4] := T_unfused[1 2 3 4; 5 6 7 8] * U[-1; 1 2] * U[-2; 3 4] *
        Udg[5 6; -3] * Udg[7 8; -4]

    return T_fused
end
