"""
$(TYPEDEF)

Bond-weighted Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T [, k=-1/2])

# Running the algorithm
    run!(::BTRG, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

# References
* [Adachi et. al. Phys. Rev. B 105 (2022)](@cite adachi2022)
"""
mutable struct BTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}, BT <: AbstractTensorMap{E, S, 1, 1}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT

    "Bond tensor on vertical bonds"
    S1::BT

    "Bond tensor on horizontal bonds"
    S2::BT

    "Bond weight exponent"
    k::Float64

    function BTRG(T::TT, k::Number) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        # Construct S1 and S2 as identity matrices.
        S1 = id(E, space(T, 2))
        S2 = id(E, space(T, 1))
        return new{E, S, TT, typeof(S1)}(T, S1, S2, k)
    end
end

# Default implementation using the optimal value for k
BTRG(T) = BTRG(T, -0.5)

function pseudopow(t::DiagonalTensorMap, a::Real; tol = eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (c, b) in blocks(t′)
        @inbounds for I in LinearAlgebra.diagind(b)
            b[I] = b[I] < tol ? b[I] : b[I]^a
        end
    end
    return t′
end

function step!(scheme::BTRG, trunc::TruncationStrategy)
    U, S, V, _ = svd_trunc(scheme.T; trunc = trunc)

    S_a = pseudopow(S, (1 - scheme.k) / 2)
    S_b = pseudopow(S, scheme.k)

    @tensor begin
        A[-1 -2; -3] := U[-1 -2; 1] * S_a[1; -3]
        B[-1; -2 -3] := S_a[-1; 1] * V[1; -2 -3]
        S1′[-1; -2] := S_b[-1; -2]
    end

    U, S, V, _ = svd_trunc(permute(scheme.T, ((3, 1), (4, 2))); trunc = trunc)

    S_a = pseudopow(S, (1 - scheme.k) / 2)
    S_b = pseudopow(S, scheme.k)

    @tensor begin
        C[-1 -2; -3] := U[-1 -2; 1] * S_a[1; -3]
        D[-1; -2 -3] := S_a[-1; 1] * V[1; -2 -3]
        S2′[-1; -2] := S_b[-1; -2]
    end

    @tensor scheme.T[-1 -2; -3 -4] := D[-1; 4 7] *
        scheme.S1[1; 7] *
        B[-2; 1 3] *
        scheme.S2[3; 2] *
        C[8 2; -4] *
        scheme.S1[8; 5] *
        A[6 5; -3] *
        scheme.S2[4; 6]
    scheme.S1 = S1′
    scheme.S2 = S2′
    return scheme
end

function Base.show(io::IO, scheme::BTRG)
    println(io, "BTRG - Bond-weighted TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * S1: $(summary(scheme.S1))")
    println(io, "  * S2: $(summary(scheme.S2))")
    println(io, "  * k: $(scheme.k)")
    return nothing
end
