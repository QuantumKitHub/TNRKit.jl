"""
$(TYPEDEF)

Corner Transfer Matrix Renormalization Group for the honeycomb lattice

### Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(T, [, symmetrize=false])

```
     (120°)
        ╲ 
         ╲ 
          ╲ 
           T -----(0°)
           ╱
          ╱
         ╱
      (240°)
```

CTM can be called with a (2, 1) tensor, where the directions are (240°, 0°, 120°) clockwise with respect to the positive x-axis.
In the flipped arrow convention, the arrows point from (120°) to (240°, 0°).
or with a (0,3) tensor (120°, 0°, 240°) where all arrows point inward (unflipped arrow convention).
The keyword argument symmetrize makes the tensor C6v symmetric when set to true. If symmetrize = false, it checks the symmetry explicitly.

### Running the algorithm
    run!(::CTM, trunc::TruncationStrategy, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Lukin et al. Phys. Rev. B 107.054424 (2023)](@cite lukin2023)
"""
mutable struct c3vCTM_honeycomb{A, S}
    T::TensorMap{A, S, 0, 3}
    C::TensorMap{A, S, 1, 1}
    L::TensorMap{A, S, 2, 1}
    R::TensorMap{A, S, 2, 1}

    function c3vCTM_honeycomb(T::TensorMap{A, S, 0, 3}) where {A, S}
        C, Ea, Eb = c3vCTM_honeycomb_init(T)

        if BraidingStyle(sectortype(T)) != Bosonic()
            @warn "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for c6vCTM"
        end
        return new{A, S}(T, C, Ea, Eb)
    end
end

function c3vCTM_honeycomb(T_flipped::TensorMap{A, S, 2, 1}; symmetrize = false) where {A, S}
    T_unflipped = permute(flip(T_flipped, [1 2]; inv = true), ((), (3, 2, 1)))
    if symmetrize
        T_unflipped = symmetrize_C3_honeycomb(T_unflipped)
    else
        !is_C3_symmetric(T_unflipped) && throw(ArgumentError("Tensor is not C3 symmetric"))
    end
    return c3vCTM_honeycomb(T_unflipped)
end

function c3vCTM_honeycomb_init(T::TensorMap{A, S, 0, 3}) where {A, S}
    S_type = scalartype(T)
    Vp = space(T)[1]'
    C = ones(S_type, oneunit(Vp) ← oneunit(Vp))
    L = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    R = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    return C, L, R
end

function Base.show(io::IO, scheme::c3vCTM_honeycomb)
    println(io, "c3vCTM_honeycomb - Corner Transfer Matrix for honeycomb lattice")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * L: $(summary(scheme.L))")
    println(io, "  * R: $(summary(scheme.R))")
    return nothing
end

# Functions to permute (unflipped) tensors under 120 degree rotation
function rotl120_pf_honeycomb(T::TensorMap{A, S, 0, 3}) where {A, S}
    return permute(T, ((), (2, 3, 1)))
end

function rotl120_pf_honeycomb(T::TensorMap{A, S, 0, 3}, i::Int) where {A, S}
    @assert i >= 0 "Negative rotation not supported"
    if i == 0
        return T
    end
    return rotl120_pf_honeycomb(rotl120_pf_honeycomb(T), i - 1)
end

function symmetrize_C3_honeycomb(T_unflipped::TensorMap{E, S, 0, 3}) where {E, S}
    return (T_unflipped + rotl120_pf_honeycomb(T_unflipped) + rotl120_pf_honeycomb(rotl120_pf_honeycomb(T_unflipped))) / 3
end

function is_C3_symmetric(T_unflipped::TensorMap{E, S, 0, 3}) where {E, S}
    return space(T_unflipped) == space(rotl120_pf_honeycomb(T_unflipped)) && norm(T_unflipped - rotl120_pf_honeycomb(T_unflipped)) < 1.0e-14
end


"""
$(TYPEDEF)

Corner Transfer Matrix Renormalization Group for the honeycomb lattice

### Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(T, [, symmetrize=false])

```
     (120°)
        ╲ 
         ╲ 
          ╲ 
           T -----(0°)
           ╱
          ╱
         ╱
      (240°)
```

CTM can be called with a (2, 1) tensor, where the directions are (240°, 0°, 120°) clockwise with respect to the positive x-axis.
In the flipped arrow convention, the arrows point from (120°) to (240°, 0°).
or with a (0,3) tensor (120°, 0°, 240°) where all arrows point inward (unflipped arrow convention).

### Running the algorithm
    run!(::CTM, trunc::TruncationStrategy, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Nyckees et al. Phys. Rev. E 108.064132 (2023)](@cite nyckees2023)
"""
mutable struct CTM_honeycomb{E, S}
    A::TensorMap{E, S, 0, 3}
    B::TensorMap{E, S, 0, 3}
    C::Vector{TensorMap{E, S, 1, 1}}
    Ta::Vector{TensorMap{E, S, 2, 1}}
    Tb::Vector{TensorMap{E, S, 2, 1}}

    function CTM_honeycomb(A::TensorMap{E, S, 0, 3}; B::TensorMap{E, S, 0, 3} = A) where {E, S}
        C, Ta, Tb = CTM_honeycomb_init(A; B)
        if BraidingStyle(sectortype(A)) != Bosonic()
            @warn "$(summary(BraidingStyle(sectortype(A)))) braiding style is not supported for c6vCTM"
        end
        return new{E, S}(A, B, C, Ta, Tb)
    end
end

function CTM_honeycomb(A_flipped::TensorMap{E, S, 2, 1}; B::TensorMap{E, S, 2, 1} = A_flipped) where {E, S}
    A_unflipped = permute(flip(A_flipped, [1 2]; inv = true), ((), (3, 2, 1)))
    B_unflipped = permute(flip(B, [1 2]; inv = true), ((), (3, 2, 1)))
    return CTM_honeycomb(A_unflipped; B = B_unflipped)
end

function CTM_honeycomb_init(A::TensorMap{E, S, 0, 3}; B::TensorMap{E, S, 0, 3} = A) where {E, S}
    S_type = scalartype(A)
    Vp = space(A)[1]'
    C = fill(ones(S_type, oneunit(Vp) ← oneunit(Vp)), 3)
    Ta = [ones(S_type, oneunit(Vp) ⊗ space(B)[mod1(dir - 1, 3)]' ← oneunit(Vp)) for dir in 1:3]
    Tb = [ones(S_type, oneunit(Vp) ⊗ space(A)[mod1(dir + 1, 3)]' ← oneunit(Vp)) for dir in 1:3]
    return C, Ta, Tb
end

function Base.show(io::IO, scheme::CTM_honeycomb)
    println(io, "CTM_honeycomb - Corner Transfer Matrix for honeycomb lattice")
    println(io, "  * A: $(summary(scheme.A))")
    println(io, "  * B: $(summary(scheme.B))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * Ta: $(summary(scheme.Ta))")
    println(io, "  * Tb: $(summary(scheme.Tb))")
    return nothing
end
