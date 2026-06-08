"""
$(TYPEDEF)

3D Higher-Order Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T)

# Running the algorithm
    run!(::HOTRG_3D, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

# References
* [Xie et. al. Phys. Rev. B 86 (2012)](@cite xie2012)

"""
mutable struct HOTRG_3D{E, S, TT <: AbstractTensorMap{E, S, 2, 4}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT

    function HOTRG_3D(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 4}}
        return new{E, S, TT}(T)
    end
end

"Twist the i-th leg of a tensor `t` if it represents a dual space."
function twistdual(t::AbstractTensorMap, is)
    is′ = filter(i -> isdual(space(t, i)), is)
    return twist(t, is′)
end

"Twist the i-th leg of a tensor `t` if it represents a non-dual space."
function twistnondual(t::AbstractTensorMap, is)
    is′ = filter(i -> !isdual(space(t, i)), is)
    return twist(t, is′)
end

function _get_MMdag_3d(
        A1::AbstractTensorMap{E, S, 2, 4}, A2::AbstractTensorMap{E, S, 2, 4}
    ) where {E, S}
    A2′ = twistnondual(A2, [2, 3, 4, 5])
    A1′ = twistnondual(A1, [1, 3, 4, 5])
    @tensoropt MM[x2 z z′ x2′] :=
        A2[z z2; Y2 X2 y2 x2] * conj(A2′[z′ z2; Y2 X2 y2 x2′])
    @tensoropt MM[x1 x2; x1′ x2′] := MM[x2 z z′ x2′] *
        A1[z1 z; Y1 X1 y1 x1] * conj(A1′[z1 z′; Y1 X1 y1 x1′])
    for ax in 1:numout(MM)
        @assert !isdual(codomain(MM, ax))
    end
    for ax in 1:numin(MM)
        @assert !isdual(domain(MM, ax))
    end
    project_hermitian!(MM)
    return MM
end

function _get_MdagM_3d(
        A1::AbstractTensorMap{E, S, 2, 4}, A2::AbstractTensorMap{E, S, 2, 4}
    ) where {E, S}
    A2′ = twistdual(A2, [2, 3, 5, 6])
    A1′ = twistdual(A1, [1, 3, 5, 6])
    @tensoropt MM[x2 z z′ x2′] :=
        conj(A2[z z2; Y2 x2 y2 X2]) * A2′[z′ z2; Y2 x2′ y2 X2]
    @tensoropt MM[x1 x2; x1′ x2′] := MM[x2 z z′ x2′] *
        conj(A1[z1 z; Y1 x1 y1 X1]) * A1′[z1 z′; Y1 x1′ y1 X1]
    for ax in 1:numout(MM)
        @assert !isdual(codomain(MM, ax))
    end
    for ax in 1:numin(MM)
        @assert !isdual(domain(MM, ax))
    end
    project_hermitian!(MM)
    return MM
end

function _get_hotrg3d_xproj(
        A1::AbstractTensorMap{E, S, 2, 4}, A2::AbstractTensorMap{E, S, 2, 4},
        trunc::TruncationStrategy
    ) where {E, S}
    # join in z-direction, keep x-indices open (A1 below A2)
    # left unitary
    MM = _get_MMdag_3d(A1, A2)
    _, U, ε = eigh_trunc!(MM; trunc)
    # right unitary
    MM = _get_MdagM_3d(A1, A2)
    _, U′, ε′ = eigh_trunc!(MM; trunc)
    return (ε > ε′) ? (U′, ε′) : (U, ε)
end

function _get_hotrg3d_yproj(
        A1::AbstractTensorMap{E, S, 2, 4}, A2::AbstractTensorMap{E, S, 2, 4},
        trunc::TruncationStrategy
    ) where {E, S}
    perm = ((1, 2), (4, 3, 6, 5))
    return _get_hotrg3d_xproj(permute(A1, perm), permute(A2, perm), trunc)
end

function _step_hotrg3d(
        A1::AbstractTensorMap{E, S, 2, 4}, A2::AbstractTensorMap{E, S, 2, 4},
        Ux::AbstractTensorMap{E, S, 2, 1}, Uy::AbstractTensorMap{E, S, 2, 1},
    ) where {E, S}
    @tensoropt T[-1 -2; -3 -4 -5 -6] :=
        conj(Ux[x1 x2; -6]) * Ux[x1′ x2′; -4] *
        conj(Uy[y1 y2; -5]) * Uy[y1′ y2′; -3] *
        A1[-1 z; y1′ x1′ y1 x1] * A2[z -2; y2′ x2′ y2 x2]
    return T
end

# HOTRG step to compress along z direction
function _step!(scheme::HOTRG_3D, trunc::TruncationStrategy)
    Ux, = _get_hotrg3d_xproj(scheme.T, scheme.T, trunc)
    Uy, = _get_hotrg3d_yproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg3d(scheme.T, scheme.T, Ux, Uy)
    return scheme
end

function step!(scheme::HOTRG_3D, trunc::TruncationStrategy)
    _step!(scheme, trunc)
    scheme.T = permute(scheme.T, ((6, 4), (2, 3, 1, 5)))
    _step!(scheme, trunc)
    scheme.T = permute(scheme.T, ((6, 4), (2, 3, 1, 5)))
    _step!(scheme, trunc)
    scheme.T = permute(scheme.T, ((6, 4), (2, 3, 1, 5)))
    return scheme
end

function Base.show(io::IO, scheme::HOTRG_3D)
    println(io, "3D HOTRG - Higher Order TRG in 3D")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
