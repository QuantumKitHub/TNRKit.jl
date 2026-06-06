const ising_βc = BigFloat(log(BigFloat(1.0) + sqrt(BigFloat(2.0))) / BigFloat(2.0))
const f_onsager::BigFloat = -2.10965114460820745966777928351108478082549327543540531781696107967700291143188081390114126499095041781
const ising_cft_exact = [
    1 / 8, 1, 9 / 8, 9 / 8, 2, 2, 2, 2, 17 / 8, 17 / 8, 17 / 8, 3, 3,
    3, 3, 3,
    25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8,
]
const ising_βc_3D = 1.0 / 4.51152469

"""
    ising_bond_tensor(β::Real, T::Type{<:Number})

Constructs the bond tensor `diag(√cosh(β), √sinh(β))` used to decompose
the Ising Boltzmann weight on a bond with effective coupling `β`.
"""
function ising_bond_tensor(β::Real, T::Type{<:Number})
    x = cosh(β)
    y = sinh(β)
    bond_matrix = T[sqrt(x) 0; 0 sqrt(y)]
    return TensorMap(bond_matrix, ℂ^2 ← ℂ^2)
end

"""
    ising_anisotropic_βc(Jx::Real, Jy::Real)

Returns the critical inverse temperature `βc` for the anisotropic 2D Ising model
on a square lattice, determined by the condition

    sinh(2 βc Jx) · sinh(2 βc Jy) = 1 .

If `Jx == Jy`, this returns the isotropic critical point `ising_βc`.
"""
function ising_anisotropic_βc(Jx::Real, Jy::Real)
    if Jx == Jy
        return Float64(ising_βc)
    end
    f(β) = sinh(2β * Jx) * sinh(2β * Jy) - 1.0
    β_max = Float64(ising_βc) / min(Jx, Jy) * 5.0
    while f(β_max) < 0
        β_max *= 2.0
    end
    return find_zero(f, (0.0, β_max))
end

"""
    f_onsager_anisotropic(β::Real, Jx::Real, Jy::Real)

Computes the exact Onsager free energy **per site** for
the 2D Ising model on a square lattice. The general formula
with anisotropic couplings `Jx`, `Jy` is

    -βf = ln(2) + 1/(8π²) ∫₀^{2π}∫₀^{2π} dθ₁dθ₂ ln[cosh(2Kx)cosh(2Ky) - sinh(2Kx)cos θ₁ - sinh(2Ky)cos θ₂]

where `Kx = β Jx`, `Ky = β Jy`.
"""
function f_onsager_anisotropic(β::Real, Jx::Real, Jy::Real)
    K1 = Float64(β * Jx)
    K2 = Float64(β * Jy)
    if Jx == Jy && abs(K1 - Float64(ising_βc)) < 1.0e-14
        return Float64(f_onsager)
    end

    c1, s1 = cosh(2K1), sinh(2K1)
    c2, s2 = cosh(2K2), sinh(2K2)
    s2_sq = s2^2

    # The 2D Onsager integral reduces to 1D after integrating out θ₂ analytically
    # (∫₀^{2π} ln(A - B cos θ) dθ = 2π ln((A + √(A²-B²))/2)):
    #
    #   -βf = ln(2) + 1/(2π) ∫₀^{π} dθ  ln((A(θ) + √(A(θ)² - s₂²)) / 2)
    #
    # where A(θ) = c₁c₂ - s₁ cos θ.
    integral, _ = quadgk(0, π) do θ
        A = c1 * c2 - s1 * cos(θ)
        log((A + sqrt(A^2 - s2_sq)) / 2)
    end

    return -(log(2.0) + integral / (2π)) / β
end

"""
    classical_ising(; kwargs...)
    classical_ising(β::Real; kwargs...)
    classical_ising(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, h = 0.0, Jx = 1.0, Jy = 1.0)
    classical_ising(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, h = 0.0, Jx = 1.0, Jy = 1.0)

Constructs the partition function tensor for a 2D square lattice
for the classical Ising model with inverse temperature `β` and external magnetic field `h`.
Compatible with no symmetry for `h ≠ 0` or with explicit ℤ₂ symmetry for `h = 0`.
Defaults to ℤ₂ symmetry and `h = 0` if the symmetry type and magnetic field are not provided.

For the anisotropic model, the coupling constants `Jx` (horizontal bonds)
and `Jy` (vertical bonds) can be specified independently.
The effective couplings are `Kx = β Jx` and `Ky = β Jy`.
Defaults to the isotropic case `Jx = Jy = 1.0`.

### Examples
```julia
    classical_ising()                           # default: ℤ₂ symmetric, isotropic at βc
    classical_ising(Trivial, 0.5; h = 1.0)     # no symmetry, with magnetic field
    classical_ising(1.0; Jx = 1.0, Jy = 0.5)   # anisotropic: Jx=1, Jy=0.5
    classical_ising(Trivial, 0.5; Jy = 0.8)     # anisotropic without symmetry

!!! info
    When studying this model with impurities, the tensor without symmetry should be constructed,
    as the impurity breaks the ℤ₂ symmetry.

!!! note "Leg convention"
    The tensor follows the standard TNRKit convention:

    ```
         3 (up,    vertical, Jy)
         |
    1 ←--+--← 4 (right, horizontal, Jx)
         |
         2 (down,  vertical, Jy)
    ```

    Legs 1 and 4 are **horizontal** bonds (coupling `Jx`),
    legs 2 and 3 are **vertical** bonds (coupling `Jy`).

See also: [`classical_ising_3D`](@ref), [`ising_anisotropic_βc`](@ref).
"""
function classical_ising(β::Real; kwargs...)
    return classical_ising(Z2Irrep, β; kwargs...)
end
classical_ising(; kwargs...) = classical_ising(ising_βc; kwargs...)
classical_ising(::Type{Trivial}; kwargs...) = classical_ising(Trivial, ising_βc; kwargs...)
function classical_ising(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, h = 0.0, Jx = 1.0, Jy = 1.0)
    Kx = β * Jx
    Ky = β * Jy
    init = zeros(T, 2, 2, 2, 2)
    for (i, j, k, l) in Iterators.product([1:2 for _ in 1:4]...)
        init[i, j, k, l] = mod(i + j + k + l, 2) == 0 ? cosh(h * β) : sinh(h * β)
    end
    init = TensorMap(init, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    bond_tensor_x = ising_bond_tensor(Kx, T)   # horizontal bonds (legs 1, 4)
    bond_tensor_y = ising_bond_tensor(Ky, T)   # vertical bonds   (legs 2, 3)

    @tensor T[-1 -2; -3 -4] := 2 * init[1 2; 3 4] * bond_tensor_x[-1; 1] * bond_tensor_y[-2; 2] * bond_tensor_y[3; -3] * bond_tensor_x[4; -4]
    return T
end
function classical_ising(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, h = 0.0, Jx = 1.0, Jy = 1.0)
    @assert h == 0.0 "External magnetic field is not compatible with ℤ₂ symmetry"
    Kx = β * Jx
    Ky = β * Jy

    xh, yh = cosh(Kx), sinh(Kx)    # horizontal bonds (legs 1, 4)
    xv, yv = cosh(Ky), sinh(Ky)    # vertical bonds   (legs 2, 3)
    w = sqrt(xh * yh * xv * yv)    # off-diagonal coupling √(cosh Kx sinh Kx cosh Ky sinh Ky)

    S = ℤ₂Space(0 => 1, 1 => 1)
    t = zeros(T, S ⊗ S ← S ⊗ S)
    block(t, Irrep[ℤ₂](0)) .= T[2 * xh * xv  2 * w;      2 * w       2 * yh * yv]
    block(t, Irrep[ℤ₂](1)) .= T[2 * w      2 * yh * xv;  2 * xh * yv   2 * w]

    return t
end

"""
    classical_ising_impurity([Type{Trivial}], β::Real; T::Type{<:Number} = Float64, h = 0.0, Jx = 1.0, Jy = 1.0)

Constructs the partition function tensor for a 2D square lattice
for the classical Ising model with a given inverse temperature `β` and external magnetic field `h`
with a magnetisation impurity. Compatible with no symmetry on each of its spaces.

### Examples
```julia
    classical_ising_impurity()                      # default: isotropic at βc
    classical_ising_impurity(0.5; h = 1.0)          # with magnetic field
    classical_ising_impurity(0.5; Jx = 1.0, Jy = 0.5)  # anisotropic couplings
```
!!! info
    When calculating the free energy with `free_energy()`, set the `initial_size` keyword argument to `2.0`.
    The initial lattice holds 2 spins.

See also: [`classical_ising`](@ref), [`classical_ising_3D`](@ref).
"""
function classical_ising_impurity(β::Real; kwargs...)
    return classical_ising_impurity(Trivial, β; kwargs...)
end
classical_ising_impurity(; kwargs...) = classical_ising_impurity(ising_βc; kwargs...)
function classical_ising_impurity(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, h = 0.0, Jx = 1.0, Jy = 1.0)
    Kx = β * Jx
    Ky = β * Jy
    init = zeros(T, 2, 2, 2, 2)
    for (i, j, k, l) in Iterators.product([1:2 for _ in 1:4]...)
        init[i, j, k, l] = mod(i + j + k + l, 2) == 0 ? sinh(h * β) : cosh(h * β)
    end
    init = TensorMap(init, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    bond_tensor_x = ising_bond_tensor(Kx, T)   # horizontal bonds (legs 1, 4)
    bond_tensor_y = ising_bond_tensor(Ky, T)   # vertical bonds   (legs 2, 3)

    @tensor t[-1 -2; -3 -4] := 2 * init[1 2; 3 4] * bond_tensor_x[-1; 1] * bond_tensor_y[-2; 2] * bond_tensor_y[3; -3] * bond_tensor_x[4; -4]
    return t
end

"""
    classical_ising_3D(; kwargs...)
    classical_ising_3D(β::Real; kwargs...)
    classical_ising_3D(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, J = 1.0)
    classical_ising_3D(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, J = 1.0)

Constructs the partition function tensor for a symmetric 3D cubic lattice
for the classical Ising model with a given inverse temperature `β`.

Compatible with no symmetry or with explicit ℤ₂ symmetry on each of its spaces.
Defaults to ℤ₂ symmetry and coupling constant `J = 1.0` if the symmetry type and coupling constant are not provided.

### Examples
```julia
    classical_ising_3D() # Default ℤ₂ symmetry, inverse temperature is `ising_βc_3D`, coupling constant is `J = 1.0`.
    classical_ising_3D(Trivial, 0.5; J = 1.5) # Custom inverse temperature and coupling constant.
    classical_ising_3D(Z2Irrep, 0.5; J = 1.5) # Custom inverse temperature and coupling constant with ℤ₂ symmetry.
```

See also: [`classical_ising`](@ref).
"""
function classical_ising_3D(β::Real; kwargs...)
    return classical_ising_3D(Z2Irrep, β; kwargs...)
end
classical_ising_3D(; kwargs...) = classical_ising_3D(ising_βc_3D; kwargs...)
classical_ising_3D(::Type{Trivial}; kwargs...) = classical_ising_3D(Trivial, ising_βc_3D; kwargs...)
function classical_ising_3D(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, J = 1.0)
    K = β * J

    # Boltzmann weights
    t = T[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(T, 2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] := O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] *
        q[-4; 4] * q[-5; 5] * q[-6; 6]

    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS)
end
function classical_ising_3D(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, J = 1.0)
    x = cosh(β * J)
    y = sinh(β * J)
    W = T[sqrt(x) sqrt(y); sqrt(x) -sqrt(y)]
    t_array = zeros(T, 2, 2, 2, 2, 2, 2)
    for (i, j, k, l, m, n) in Iterators.product([1:2 for _ in 1:6]...)
        for a in 1:2
            # Outer product of W[a, :] with itself 6 times
            t_array[i, j, k, l, m, n] += W[a, i] * W[a, j] * W[a, k] * W[a, l] * W[a, m] *
                W[a, n]
        end
    end
    S = ℤ₂Space(0 => 1, 1 => 1)
    t = TensorMap(t_array, S ⊗ S ⊗ S ← S ⊗ S ⊗ S)

    return permute(t, ((1, 4), (5, 6, 2, 3)))
end
