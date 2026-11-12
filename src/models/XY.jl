function algebraic_initialization(m::AbstractTensorMap{E, S, 1, 2}, bond::AbstractTensorMap{E, S, 1, 1}) where {E, S}
    @tensor opt = true T[l u; d r] :=
        m[u; Au Bu] *
        bond[Au; Ad] *
        bond[Bu; Bd] *
        m[Bd; Cu r] *
        m'[l Ad; Du] *
        bond[Du; Dd] *
        bond[Cu; Cd] *
        m'[Dd Cd; d]
    return T
end

const XY_βc = 1.1199 # This is an approximation!

"""
    classical_XY(charge_trunc::Int; kwargs...)
    classical_XY(beta::Float64, charge_trunc::Int; kwargs...)
    classical_XY(::Type{U1Irrep}, beta::Float64, charge_trunc::Int; T::Type{<:Number} = Float64)
    classical_XY(::Type{CU1Irrep}, beta::Float64, charge_trunc::Int; T::Type{<:Number} = Float64)

Constructs the partition function tensor for a symmetric 2D square lattice
for the classical XY model using inverse temperature `beta`
and charge truncation `charge_trunc`.
Compatible with U(1) symmetry or CU(1) = O(2) symmetry on each of its spaces.
Defaults to CU(1) symmetry if the symmetry type is not provided.

# Examples
```julia
    classical_XY(U1Irrep, 0.9, 6)
    classical_XY(CU1Irrep, 0.9, 4)
```

# References
* [Yu et. al. 10.1103/PhysRevE.89.013308 (2014)](@cite Yu_2014)
"""
function classical_XY(beta::Float64, charge_trunc::Int; kwargs...)
    return classical_XY(CU1Irrep, beta, charge_trunc; kwargs...)
end
classical_XY(charge_trunc::Int; kwargs...) = classical_XY(XY_βc, charge_trunc; kwargs...)
classical_XY(::Type{U1Irrep}, charge_trunc::Int; kwargs...) = classical_XY(U1Irrep, XY_βc, charge_trunc; kwargs...)
function classical_XY(::Type{U1Irrep}, beta::Float64, charge_trunc::Int; T::Type{<:Number} = Float64)
    FunU1 = U1Space(map(x -> (x => 1), (-charge_trunc):charge_trunc))

    m = ones(T, FunU1 ← FunU1 ⊗ FunU1)

    bond = zeros(T, FunU1 ← FunU1)
    for (s, f) in fusiontrees(bond)
        charge = s.uncoupled[1].charge
        bond[s, f] .= besseli(charge, beta)
    end

    return algebraic_initialization(m, bond)
end
function classical_XY(::Type{CU1Irrep}, beta::Float64, charge_trunc::Int; T::Type{<:Number} = Float64)
    FunU1_0 = CU1Space((0, 0) => 1)
    FunU1_1 = CU1Space(((i, 2) => 1 for i in 1:charge_trunc))
    FunU1 = FunU1_0 ⊕ FunU1_1

    m = zeros(T, FunU1 ← FunU1 ⊗ FunU1)
    for (to, from) in fusiontrees(m)
        left, right = from.uncoupled
        if (left == right) && !isunit(left) && isunit(from.coupled)
            m[to, from] .= sqrt(2)
        else
            m[to, from] .= 1
        end
    end

    bond = zeros(T, FunU1 ← FunU1)

    for (s, f) in fusiontrees(bond)
        charge = s.uncoupled[1].j
        bond[s, f] .= besseli(charge, beta)
    end

    return algebraic_initialization(m, bond)
end
