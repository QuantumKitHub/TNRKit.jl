#     ┌──┐    ┌────┐
#     │  │    │    │
# ───►│E1│───►│ C2 │
#     └──┘    └──┬─┘
#       ▲        │
#       │        ▼
#              ┌──┐
#              │  │
#           ──►│E2│
#              └─┬┘
#                │
#                ▼

"""
$(TYPEDEF)

Reflection-symmetric Corner Transfer Matrix

# Constructors
    $(FUNCTIONNAME)(T)

CTM with spatial reflection symmetry along x and y axis. This allows different edge tensors for the vertical and horizontal edges.
The corner tensors are related by their mirror images.

# Running the algorithm
    run!(::rCTM, trunc::TruncationStrategy, criterion::TNRKit.stopcrit[, verbosity=1])

# Fields

$(TYPEDFIELDS)
"""
mutable struct rCTM{E, S, TT <: AbstractTensorMap{E, S, 2, 2}, TC <: AbstractTensorMap{E, S, 1, 1}, TE <: AbstractTensorMap{E, S, 2, 1}} <: TNRScheme{E, S}
    T::TT
    C2::TC
    E1::TE
    E2::TE

    function rCTM(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        if !(E <: Real)
            @error "This scheme only support tensors with real numbers"
        end
        C, E1, E2 = rCTM_init(T)
        @assert BraidingStyle(sectortype(T)) == Bosonic() "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for rCTM"
        return new{E, S, TT, typeof(C), typeof(E1)}(T, C, E1, E2)
    end
end

function rCTM_init(T)
    elt = typeof(T.data[1])
    Vp1 = space(T)[3]'
    Vp2 = space(T)[4]'
    C = ones(elt, oneunit(Vp1) ← oneunit(Vp2))
    E1 = ones(elt, oneunit(Vp1) ⊗ Vp1 ← oneunit(Vp1))
    E2 = ones(elt, oneunit(Vp2) ⊗ Vp2 ← oneunit(Vp2))
    return C, E1, E2
end

function rt_build_corner_matrix(scheme::rCTM)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.E1[-1 3; 1] * scheme.C2[1; 2] *
        scheme.E2[2 4; -3] * scheme.T[-2 -4; 3 4]
    return mat
end

function find_UVt(scheme, trunc)
    mat = rt_build_corner_matrix(scheme)
    U, S, Vt = svd_trunc(mat; trunc = trunc & trunctol(atol = 1.0e-20))
    return mat, U, S, Vt
end

function step!(scheme::rCTM, trunc)
    mat, U, S, Vt = find_UVt(scheme, trunc)

    scheme.C2 = adjoint(U) * mat * adjoint(Vt)
    @tensor opt = true scheme.E1[-1 -2; -3] := scheme.E1[1 5; 3] * scheme.T[2 -2; 5 4] *
        U[3 4; -3] * conj(U[1 2; -1])
    @tensor opt = true scheme.E2[-1 -2; -3] := scheme.E2[1 5; 3] * scheme.T[-2 4; 2 5] *
        conj(Vt[-3; 3 4]) * Vt[-1; 1 2]

    normalize!(scheme.C2)
    normalize!(scheme.E1)
    normalize!(scheme.E2)
    # symmetrization of the edges to avoid the breaking of reflection due to the numerical error
    scheme.E1 = (scheme.E1 + flip(permute(scheme.E1, ((3, 2), (1,))), (1, 3)) / 2)
    scheme.E2 = (scheme.E2 + flip(permute(scheme.E1, ((3, 2), (1,))), (1, 3)) / 2)
    return S
end

function run!(
        scheme::rCTM, trunc::TruncationStrategy,
        criterion::TNRKit.stopcrit; verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        S_prev = id(domain(scheme.C2))

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), ε = $(ε)"

            S = step!(scheme, trunc)

            if space(S) == space(S_prev)
                ε = norm(S^4 - S_prev^4)
            end

            S_prev = S

            steps += 1
            crit = criterion(steps, ε)
        end

        @infov 1 "Simulation finished\n $(TNRKit.stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function lnz(scheme::rCTM)
    corners = [adjoint(scheme.C2), scheme.C2, adjoint(scheme.C2), scheme.C2]
    edges = [scheme.E1 scheme.E2 flip(scheme.E1, 2) flip(scheme.E2, 2)]

    return real(log(network_value(scheme.T, corners, edges)))
end

function Base.show(io::IO, scheme::rCTM)
    println(io, "rCTM - reflection symmetric Corner Transfer Matrix")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * E1: $(summary(scheme.E1))")
    println(io, "  * E2: $(summary(scheme.E2))")
    println(io, "  * C: $(summary(scheme.C2))")
    return nothing
end
