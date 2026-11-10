const ising_βc_honeycomb = BigFloat(BigFloat(asinh(BigFloat(sqrt(BigFloat(3.0))))) / BigFloat(2.0))
const f_onsager_honeycomb::BigFloat = -1.556707467816387475214957698255679494804

"""
    classical_ising_honeycomb(; kwargs...)
    classical_ising_honeycomb(β::Real; kwargs...)
    classical_ising_honeycomb(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64)
    classical_ising_honeycomb(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64)

Constructs the partition function tensor for a 2D honeycomb lattice
for the classical Ising model with a given inverse temperature `β`.
Compatible with no symmetry or with explicit ℤ₂ symmetry on each of its spaces.
Defaults to ℤ₂ symmetry and inverse temperature `ising_βc_honeycomb` if the symmetry type and inverse temperature are not provided.

# Examples
```julia
    classical_ising_honeycomb() # Default ℤ₂ symmetry, inverse temperature is `ising_βc_honeycomb`
    classical_ising_honeycomb(Trivial, 0.5) # Custom inverse temperature.
    classical_ising_honeycomb(Z2Irrep, 0.5) # Custom inverse temperature with explicit ℤ₂ symmetry.
```

"""
function classical_ising_honeycomb(β::Real; kwargs...)
    return classical_ising_honeycomb(Z2Irrep, β; kwargs...)
end
classical_ising_honeycomb(; kwargs...) = classical_ising_honeycomb(ising_βc_honeycomb; kwargs...)
classical_ising_honeycomb(::Type{Trivial}; kwargs...) = classical_ising_honeycomb(Trivial, ising_βc_honeycomb; kwargs...)
function classical_ising_honeycomb(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64)
    t = T[exp(β) exp(-β); exp(-β) exp(β)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(T, 2, 2, 2)
    O[1, 1, 1] = one(T)
    O[2, 2, 2] = one(T)

    H = [1 1; 1 -1] / sqrt(2)

    @tensor o[-1 -2 -3] := O[1 2 3] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3]
    @tensor o2[-1 -2 -3] := o[1 2 3] * H[-1; 1] * H[-2; 2] * H[-3; 3]

    return TensorMap(o2, ℂ^2 * ℂ^2, ℂ^2)
end
function classical_ising_honeycomb(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    tens = zeros(T, S ⊗ S ← S)

    block(tens, Irrep[ℤ₂](0)) .= [2 * x^(3 / 2); 2 * sqrt(x) * y;;]
    block(tens, Irrep[ℤ₂](1)) .= [2 * sqrt(x) * y; 2 * sqrt(x) * y;;]
    return tens
end
