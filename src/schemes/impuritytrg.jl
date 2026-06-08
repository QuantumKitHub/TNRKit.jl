"""
$(TYPEDEF)

Impurity method for Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T, T_imp1, T_imp2, T_imp3, T_imp4)

# Running the algorithm
    run!(::ImpurityTRG, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=ImpurityTRG_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

Illustration (with p the pure tensor, and the p fills in the rest of the lattice):
```
    p   p
    |   |  
p---1---2---p
    |   |
p---4---3---p
    |   |
    p   p
```

# References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)
* [Morita et. al. Phys. Rev. B 111 (2025)](@cite morita2025)
"""
mutable struct ImpurityTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "Pure tensor"
    T::TT

    "Impurity tensor on lattice site 1"
    T_imp1::TT

    "Impurity tensor on lattice site 2"
    T_imp2::TT

    "Impurity tensor on lattice site 3"
    T_imp3::TT

    "Impurity tensor on lattice site 4"
    T_imp4::TT

    function ImpurityTRG(T::TT, T_imp1::TT, T_imp2::TT, T_imp3::TT, T_imp4::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        @assert space(T, 1) == space(T_imp1, 1) == space(T_imp2, 1) == space(T_imp3, 1) == space(T_imp4, 1) "First space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"
        @assert space(T, 2) == space(T_imp1, 2) == space(T_imp2, 2) == space(T_imp3, 2) == space(T_imp4, 2) "Second space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"
        @assert space(T, 3) == space(T_imp1, 3) == space(T_imp2, 3) == space(T_imp3, 3) == space(T_imp4, 3) "Third space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"
        @assert space(T, 4) == space(T_imp1, 4) == space(T_imp2, 4) == space(T_imp3, 4) == space(T_imp4, 4) "Fourth space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"
        return new{E, S, TT}(T, T_imp1, T_imp2, T_imp3, T_imp4)
    end
end

function step!(scheme::ImpurityTRG, trunc::TruncationStrategy)
    # Tensor1
    A1, B1 = SVD12(scheme.T_imp1, trunc)

    # Tensor2
    tensor2p = transpose(scheme.T_imp2, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(scheme.T_imp3, trunc)

    # Tensor4
    tensor4p = transpose(scheme.T_imp4, ((2, 4), (1, 3)))
    C4, D4 = SVD12(tensor4p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(scheme.T, trunc)
    tensorpurep = transpose(scheme.T, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Contract
    @planar scheme.T[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T_imp1[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T_imp2[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T_imp3[-1, -2; -3, -4] := D4[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * A3[3 2; -4]
    @planar scheme.T_imp4[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C4[4 3; -3] * A1[3 2; -4]

    return scheme
end

function Base.show(io::IO, scheme::ImpurityTRG)
    println(io, "ImpurityTRG - Impurity TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * T_imp1: $(summary(scheme.T_imp1))")
    println(io, "  * T_imp2: $(summary(scheme.T_imp2))")
    println(io, "  * T_imp3: $(summary(scheme.T_imp3))")
    println(io, "  * T_imp4: $(summary(scheme.T_imp4))")
    return nothing
end

run!(scheme::ImpurityTRG, trscheme::TruncationStrategy, criterion::stopcrit; kwargs...) = run!(scheme, trscheme, criterion, ImpurityTRG_Finalizer; kwargs...)
