"""
$(TYPEDEF)

Corner Transfer Matrix Renormalization Group for the triangular lattice

# Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(T, [, symmetrize=false])

```
     (120°)     (60°)
        ╲       ╱
         ╲     ╱
          ╲   ╱
(180°)----- T -----(0°)
           ╱ ╲
          ╱   ╲
         ╱     ╲
      (240°) (300°)
```

CTM can be called with a (3, 3) tensor, where the directions are (180°, 240°, 300°, 120°, 60°, 0°) clockwise with respect to the positive x-axis.
In the flipped arrow convention, the arrows point from (120°, 60°, 0°) to (180°, 240°, 300°).
or with a (0,6) tensor (120°, 60°, 0°, 300°, 240°, 180°) where all arrows point inward (unflipped arrow convention).
The keyword argument symmetrize makes the tensor C6v symmetric when set to true. If symmetrize = false, it checks the symmetry explicitly.

# Running the algorithm
    run!(::CTM, trunc::TruncationStrategy, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

# References
* [Naumann et al. Phys. Rev. B 113(4) (2026)](@cite naumann2026)
"""
mutable struct CTM_triangular{A, S}
    T::TensorMap{A, S, 0, 6}
    C::Array{TensorMap{A, S, 2, 1}, 1}
    Ea::Array{TensorMap{A, S, 2, 1}, 1}
    Eb::Array{TensorMap{A, S, 2, 1}, 1}

    function CTM_triangular(T::TensorMap{A, S, 0, 6}; vspace = oneunit(space(T)[1]')) where {A, S}
        C, Ea, Eb = CTM_triangular_init(T, vspace)

        if BraidingStyle(sectortype(T)) != Bosonic()
            @warn "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for c6vCTM"
        end
        return new{A, S}(T, C, Ea, Eb)
    end
end

function CTM_triangular(T_flipped::TensorMap{A, S, 3, 3}; vspace = oneunit(space(T_flipped)[4]'), symmetrize = false) where {A, S}
    T_unflipped = permute(flip(T_flipped, (1, 2, 3); inv = true), ((), (4, 5, 6, 3, 2, 1)))

    if symmetrize
        T_unflipped = symmetrize_C6v(T_unflipped)
    end
    return CTM_triangular(T_unflipped; vspace)
end

function CTM_triangular_init(T::TensorMap{A, S, 0, 6}, vspace) where {A, S}
    S_type = scalartype(T)
    Vp = space(T)[1]'
    C = fill(ones(S_type, vspace ⊗ Vp ← vspace), 6)
    Ea = fill(ones(S_type, vspace ⊗ Vp ← vspace), 6)
    Eb = fill(ones(S_type, vspace ⊗ Vp ← vspace), 6)
    return C, Ea, Eb
end

"""
$(TYPEDEF)

C6v symmetric Corner Transfer Matrix Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(T, [, symmetrize=false])

```
     (120°)     (60°)
        ╲       ╱
         ╲     ╱
          ╲   ╱
(180°)----- T -----(0°)
           ╱ ╲
          ╱   ╲
         ╱     ╲
      (240°) (300°)
```

c6vCTM can be called with a (3, 3) tensor, where the directions are (180°, 240°, 300°, 120°, 60°, 0°) clockwise with respect to the positive x-axis.
In the flipped arrow convention, the arrows point from (120°, 60°, 0°) to (180°, 240°, 300°).
or with a (0,6) tensor (120°, 60°, 0°, 300°, 240°, 180°) where all arrows point inward (unflipped arrow convention).
The keyword argument symmetrize makes the tensor C6v symmetric when set to true. If symmetrize = false, it checks the symmetry explicitly.

# Running the algorithm
    run!(::c6vCTM, trunc::MatrixAlgebraKit.TruncationStrategy, stop::Stopcrit[, finalize_beginning=true, projectors=:twothirds, conditioning=true, verbosity=1])

`projectors` can either be :twothirds or :full, determining the type of projectors used in the renormalization step. This is based on https://arxiv.org/abs/2510.04907v1.
`conditioning` determines whether to condition the second projector construction. This is based on https://doi.org/10.1103/PhysRevB.98.235148.

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

# References
* [Naumann et al. Phys. Rev. B 113(4) (2026)](@cite naumann2026)
* c6v implementation provided by Sander De Meyer
"""
mutable struct c6vCTM_triangular{A, S}
    T::TensorMap{A, S, 0, 6}
    C::TensorMap{A, S, 2, 1}
    Ea::TensorMap{A, S, 2, 1}
    Eb::TensorMap{A, S, 2, 1}

    function c6vCTM_triangular(T::TensorMap{A, S, 0, 6}) where {A, S}
        C, Ea, Eb = c6vCTM_triangular_init(T)

        if BraidingStyle(sectortype(T)) != Bosonic()
            @warn "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for c6vCTM"
        end
        return new{A, S}(T, C, Ea, Eb)
    end
end

function c6vCTM_triangular(T_flipped::TensorMap{A, S, 3, 3}; symmetrize = false) where {A, S}
    T_unflipped = permute(flip(T_flipped, (1, 2, 3); inv = true), ((), (4, 5, 6, 3, 2, 1)))

    if symmetrize
        T_unflipped = symmetrize_C6v(T_unflipped)
    else
        @assert norm(T_flipped - T_flipped') < 1.0e-14 "Tensor is not hermitian. Error = $(norm(T_flipped - T_flipped'))"
        @assert norm(T_unflipped - rotl60_pf(T_unflipped)) < 1.0e-14 "Tensor is not C6 symmetric. Error = $(norm(T_unflipped - rotl60_pf(T_unflipped)))"
    end
    return c6vCTM_triangular(T_unflipped)
end

# Functions to permute (flipped and unflipped) tensors under 60 degree rotation
function rotl60_pf(T::TensorMap{A, S, 3, 3}) where {A, S}
    return permute(T, ((4, 1, 2), (5, 6, 3)))
end

function rotl60_pf(T::TensorMap{A, S, 0, 6}) where {A, S}
    return permute(T, ((), (2, 3, 4, 5, 6, 1)))
end

# Function to construct a C6v symmetric tensor from a given tensor in the unflipped arrow convention
function symmetrize_C6v(T_unflipped)
    T_c4_unflipped = (
        T_unflipped + rotl60_pf(T_unflipped) + rotl60_pf(rotl60_pf(T_unflipped)) + rotl60_pf(rotl60_pf(rotl60_pf(T_unflipped))) +
            rotl60_pf(rotl60_pf(rotl60_pf(rotl60_pf(T_unflipped)))) + rotl60_pf(rotl60_pf(rotl60_pf(rotl60_pf(rotl60_pf(T_unflipped)))))
    ) / 6
    T_c4_flipped = permute(flip(T_c4_unflipped, (4, 5, 6); inv = false), ((6, 5, 4), (1, 2, 3)))
    T_c4v_flipped = (T_c4_flipped + T_c4_flipped') / 2
    T_c4v_unflipped = permute(flip(T_c4v_flipped, (1, 2, 3); inv = true), ((), (4, 5, 6, 3, 2, 1)))
    return T_c4v_unflipped
end

function c6vCTM_triangular_init(T::TensorMap{A, S, 0, 6}) where {A, S}
    S_type = scalartype(T)
    Vp = space(T)[1]'
    C = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    Ea = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    Eb = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    return C, Ea, Eb
end
