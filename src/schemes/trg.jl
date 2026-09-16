"""
$(TYPEDEF)

State holding a single tensor for a 1-site unit cell.

# Constructors
    $(FUNCTIONNAME)(T)


# Fields

$(TYPEDFIELDS)
"""
mutable struct OneSiteState{TT <: AbstractTensorMap{<:Any, <:Any, 2, 2}}
    "central tensor"
    T::TT

    function OneSiteState(T::TT) where {TT <: AbstractTensorMap{<:Any, <:Any, 2, 2}}
        return new{TT}(T)
    end
end

"""
$(TYPEDEF)

Tensor Renormalization Group algorithm.

Each step rescales the lattice by a (linear) factor of √2,
and rotates the lattice by 45 degrees in counter clockwise direction.
The elementary modular parameter `τ₀ ↦ (τ₀ - 1) / (τ₀ + 1)`.

# Constructors

All parameters are passed as keyword arguments with sensible defaults:

    $(FUNCTIONNAME)(; trunc=truncrank(16), stop=maxiter(20))

# Fields

$(TYPEDFIELDS)

# References
* [Levin & Nave Phys. Rev. Letters 99(12) (2007)](@cite levin2007)
"""
Base.@kwdef struct TRG <: TNRAlgorithm
    "Truncation strategy for SVD steps"
    trunc::TruncationStrategy = truncrank(16)
    "Stopping criterion"
    stop::stopcrit = maxiter(20)
end

function Renormalizer(alg::TRG, T::TT) where {TT}
    state = OneSiteState(T)
    return Renormalizer(alg, state, scalartype(T)[])
end

function Renormalizer(alg::TRG, state::OneSiteState)
    return Renormalizer(alg, state, scalartype(state.T)[])
end

function step!(state::OneSiteState, alg::TRG)
    A, B = SVD12(state.T, alg.trunc)
    Tp = transpose(state.T, ((2, 4), (1, 3)))
    C, D = SVD12(Tp, alg.trunc)
    @plansor state.T[-1 -2; -3 -4] := D[-2; 1 2] * B[-1; 4 1] * C[4 3; -3] * A[3 2; -4]
    return state
end

function Base.show(io::IO, alg::TRG)
    println(io, "TRG")
    println(io, "  * truncation: $(alg.trunc)")
    println(io, "  * stop: $(alg.stop)")
    return nothing
end
