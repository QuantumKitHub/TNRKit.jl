"""
    sixvertex(; kwargs...)
    sixvertex(::Type{Trivial}; T::Type{<:Number} = Float64, a = 1.0, b = 1.0, c = 1.0)
    sixvertex(::Type{U1Irrep}; T::Type{<:Number} = Float64, a = 1.0, b = 1.0, c = 1.0)
    sixvertex(::Type{CU1Irrep}; T::Type{<:Number} = Float64, a = 1.0, b = 1.0, c = 1.0)

Constructs the partition function tensor for the six-vertex model with a given symmetry type and coupling constants `a`, `b`, and `c`.
Compatible with no symmetry, U(1) symmetry, or CU(1) symmetry on each of its spaces.

# Defaults
    - T: Float64
    - symmetry: CU1Irrep
    - a: 1.0
    - b: 1.0
    - c: 1.0

# Examples
```julia
    sixvertex() # Default symmetry is `CU1Irrep`, coupling constants are `a = 1.0`, `b = 1.0`, `c = 1.0`.
    sixvertex(Trivial) # No symmetry with default coupling constants.
    sixvertex(ComplexF64, U1Irrep; a = 2.0, b = 3.0, c = 4.0) # U1 symmetry with custom coupling constants and element type.
```

Note: The free energy density depends on the boundary conditions.  
"""
function sixvertex(; kwargs...)
    return sixvertex(CU1Irrep; kwargs...)
end
function sixvertex(::Type{Trivial}; T::Type{<:Number} = Float64, a = 1.0, b = 1.0, c = 1.0)
    d = T[
        a 0 0 0
        0 c b 0
        0 b c 0
        0 0 0 a
    ]
    return TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
end
function sixvertex(::Type{U1Irrep}; T::Type{<:Number} = Float64, a = 1.0, b = 1.0, c = 1.0)
    pspace = U1Space(-1 // 2 => 1, 1 // 2 => 1)
    mpo = zeros(T, pspace ⊗ pspace, pspace ⊗ pspace)
    block(mpo, Irrep[U₁](0)) .= [b c; c b]
    block(mpo, Irrep[U₁](1)) .= reshape([a], (1, 1))
    block(mpo, Irrep[U₁](-1)) .= reshape([a], (1, 1))
    return mpo
end
function sixvertex(::Type{CU1Irrep}; T::Type{<:Number} = Float64, a = 1.0, b = 1.0, c = 1.0)
    pspace = CU1Space(1 // 2 => 1)
    mpo = zeros(T, pspace ⊗ pspace, pspace ⊗ pspace)
    block(mpo, Irrep[CU₁](0, 0)) .= reshape([b + c], (1, 1))
    block(mpo, Irrep[CU₁](0, 1)) .= reshape([-b + c], (1, 1))
    block(mpo, Irrep[CU₁](1, 2)) .= reshape([a], (1, 1))
    return mpo
end
