"""
    weyl_heisenberg_matrices(dimension [, eltype])

the Weyl-Heisenberg matrices according to [Wikipedia](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices#Sylvester's_generalized_Pauli_matrices_(non-Hermitian)).
"""
function weyl_heisenberg_matrices(q::Int, elt::Type{<:Number} = ComplexF64)
    U = zeros(elt, q, q) # clock matrix
    V = zeros(elt, q, q) # shift matrix
    W = zeros(elt, q, q) # DFT
    ω = cis(2 * pi / q)

    for row in 1:q
        U[row, row] = ω^(row - 1)
        V[row, mod1(row - 1, q)] = one(elt)
        for col in 1:q
            W[row, col] = ω^((row - 1) * (col - 1))
        end
    end
    return U, V, W / elt(sqrt(q))
end

function potts_tensor(q::Int, β::Real; T::Type{<:Number} = Float64)
    A_potts = zeros(T, q, q, q, q)
    for i in 1:q, j in 1:q, k in 1:q, l in 1:q
        E = -(Int(i == j) + Int(j == l) + Int(k == l) + Int(k == i))
        A_potts[i, j, k, l] = exp(-β * E)
    end
    P = ℂ^q
    return TensorMap(A_potts, P ⊗ P ← P ⊗ P)
end

"""
$(SIGNATURES)

returns the inverse critical temperature for the classical q-state Potts model on a 2D square lattice.

See also: [`classical_potts`](@ref).
"""
potts_βc(q) = log(1.0 + sqrt(q))

"""
    classical_potts(q::Int, β::Real; kwargs...)
    classical_potts(::Type{Trivial}, q::Int, β::Float64; kwargs...)
    classical_potts(::Type{ZNIrrep{N}}, q::Int, β::Float64; kwargs...) where {N}

Constructs the partition function tensor for the classical Potts model with `q` states
and a given inverse temperature `β`.

Compatible with no symmetry or with explicit ℤq symmetry on each of its spaces.
Defaults to ℤq symmetry if the symmetry type is not provided.

# Examples
```julia
    classical_potts(3) # Default has Z₃ symmetry and uses `potts_βc(3)` as the inverse temperature.
    classical_potts(Z3Irrep, 3, 0.5) # Custom inverse temperature with explicit ℤ₃ symmetry.
```

!!! info
    When studying this model with impurities, the tensor without symmetry should be constructed, as the impurity breaks the ℤq symmetry.

See also: [`potts_βc`](@ref).
"""
function classical_potts(q::Int, β::Real; kwargs...)
    return classical_potts(ZNIrrep{q}, q, β; kwargs...)
end
classical_potts(::Type{Trivial}, q::Int64; kwargs...) = classical_potts(Trivial, q, potts_βc(q); kwargs...)
classical_potts(q::Int; kwargs...) = classical_potts(q, potts_βc(q); kwargs...)
function classical_potts(::Type{Trivial}, q::Int, β::Real; T::Type{<:Number} = Float64)
    return potts_tensor(q, β; T = T)
end
function classical_potts(::Type{ZNIrrep{N}}, q::Int64, β::Real; T::Type{<:Number} = Float64) where {N}
    @assert N == q "number of irreps must match the number of states"
    A = potts_tensor(q, β; T = T)

    _, _, W = weyl_heisenberg_matrices(q, complex(T))
    P = TensorMap(W, ℂ^q ← ℂ^q)
    Udat = reshape(((P' ⊗ P') * A * (P ⊗ P)).data, (q, q, q, q))
    Vp = Vect[ZNIrrep{q}](sector => 1 for sector in 0:(q - 1))
    A_potts = TensorMap(Udat, Vp ⊗ Vp ← Vp ⊗ Vp)
    return T <: Real ? real(A_potts) : A_potts
end

"""
    classical_potts_impurity(q::Int64, β::Real, k1::Int64 = 1, k2::Int64 = 1; T::Type{<:Number} = Float64)
    classical_potts_impurity(q::Int64, k1::Int64, k2::Int64; T::Type{<:Number} = Float64)
    classical_potts_impurity(q::Int64; T::Type{<:Number} = Float64)

Constructs the partition function tensor for a Potts model with `q` states
and a given inverse temperature `β` with impurities in sectors `k1` and `k2`.

The impurity breaks the ℤq symmetry, but the impurity sectors match the symmetry sectors of the model.
# Examples
```julia
    classical_potts_impurity(3) # Default inverse temperature is `potts_βc(3)`
    classical_potts_impurity(3, 1, 2) # Custom inverse temperature and impurity sectors.
    classical_potts_impurity(3, 0.5, 2, 3) # Custom inverse temperature and impurity sectors.
```

See also: [`classical_potts`](@ref), [`potts_βc`](@ref).
"""
function classical_potts_impurity(q::Int64, β::Real, k1::Int64 = 1, k2::Int64 = 1; kwargs...)
    return classical_potts_impurity(Trivial, q, β, k1, k2; kwargs...)
end
classical_potts_impurity(q::Int64, k1::Int64, k2::Int64; kwargs...) = classical_potts_impurity(q, potts_βc(q), k1, k2; kwargs...)
classical_potts_impurity(q::Int64; kwargs...) = classical_potts_impurity(q, potts_βc(q), 1, 1; kwargs...)
function classical_potts_impurity(::Type{Trivial}, q::Int64, β::Real, k1::Int64 = 1, k2::Int64 = 1; T::Type{<:Number} = Float64)
    bond_tensor = zeros(T, q, q)
    for i in 0:(q - 1)
        bond_tensor[i + 1, i + 1] = sqrt(exp(β) - 1 + q * (i == 0))
    end
    Vp = ℂ^q
    bond_tensor = TensorMap(bond_tensor, Vp ← Vp)

    core_tensor = zeros(T, q, q, q, q)
    for (i, j, k, l) in Iterators.product(0:(q - 1), 0:(q - 1), 0:(q - 1), 0:(q - 1))
        core_tensor[i + 1, j + 1, k + 1, l + 1] =
            mod(i + j - k - l + k1 - k2, q) == 0 ? 1 : 0
    end

    core_tensor = TensorMap(core_tensor, Vp ⊗ Vp ← Vp ⊗ Vp)

    @tensor t[-1 -2; -3 -4] := core_tensor[1 2; 3 4] * bond_tensor[-1; 1] * bond_tensor[-2; 2] * bond_tensor[3; -3] * bond_tensor[4; -4] * (1 / q)

    return t
end
