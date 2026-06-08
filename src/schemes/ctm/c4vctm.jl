"""
$(TYPEDEF)

C4v symmetric Corner Transfer Matrix Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(T, [, symmetrize=false])

c4vCTM can be called with a (2,2) tensor (West, South, North, East) with the usual arrow conventions (flipped arrow convention), 
or with a (0,4) tensor (North, East, South, West) (unflipped arrow convention).
The keyword argument symmetrize makes the tensor C4v symmetric when set to true. If symmetrize = false, it checks the symmetry explicitly.

# Running the algorithm
    run!(::c4vCTM, trunc::TruncationStrategy, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)
"""
mutable struct c4vCTM{A, S, TT <: AbstractTensorMap{A, S, 0, 4}, TC <: AbstractTensorMap{A, S, 1, 1}, TE <: AbstractTensorMap{A, S, 2, 1}} <: TNRScheme{A, S}
    "Central tensor"
    T::TT

    "Corner tensor"
    C::TC

    "Edge tensor"
    E::TE

    function c4vCTM(T::TT) where {A, S, TT <: AbstractTensorMap{A, S, 0, 4}}
        C, E = c4vCTM_init(T)

        return new{A, S, TT, typeof(C), typeof(E)}(T, C, E)
    end
end

function c4vCTM(T_flipped::TensorMap{A, S, 2, 2}; symmetrize = false) where {A, S}
    T_unflipped = permute(flip(T_flipped, (1, 2); inv = true), ((), (3, 4, 2, 1)))
    if symmetrize
        T_unflipped = symmetrize_C4v(T_unflipped)
    else
        @assert norm(T_flipped - T_flipped') < 1.0e-14
        @assert norm(T_unflipped - rotl90_pf(T_unflipped)) < 1.0e-14
    end
    return c4vCTM(T_unflipped)
end

# Functions to permute (flipped and unflipped) tensors under 90 degree rotation
function rotl90_pf(T::TensorMap{A, S, 2, 2}) where {A, S}
    return permute(T, ((3, 1), (4, 2)))
end

function rotl90_pf(T::TensorMap{A, S, 0, 4}) where {A, S}
    return permute(T, ((), (2, 3, 4, 1)))
end

# Function to construct a C4v symmetric tensor from a given tensor in the unflipped arrow convention
function symmetrize_C4v(T_unflipped)
    T_c4_unflipped = (T_unflipped + rotl90_pf(T_unflipped) + rotl90_pf(rotl90_pf(T_unflipped)) + rotl90_pf(rotl90_pf(rotl90_pf(T_unflipped)))) / 4
    T_c4_flipped = permute(flip(T_c4_unflipped, (3, 4); inv = false), ((4, 3), (1, 2)))
    T_c4v_flipped = (T_c4_flipped + T_c4_flipped') / 2
    T_c4v_unflipped = permute(flip(T_c4v_flipped, (1, 2); inv = true), ((), (3, 4, 2, 1)))
    return T_c4v_unflipped
end

# Below, I wrote a code with the following correspondence. (O,C,T) <=> (scheme.T, scheme.C, scheme.E)
# https://www.issp.u-tokyo.ac.jp/public/caqmp2019/slides/808L_Okubo.pdf
#=
┌───────┐       ┌───────┐       
│       │       │       │       
│       │       │       │       
│   C   ├──────►│   E   ├──────►
│       │       │       │       
└───────┘       └───────┘       
    ▲               ▲           
    │               │           
    │               │           
=#

function run!(
        scheme::c4vCTM, trunc::TruncationStrategy, criterion::stopcrit;
        verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        S_prev = id(domain(scheme.C))

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

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function step!(scheme::c4vCTM, trunc)
    mat, U, S = find_U_sym(scheme, trunc)

    @tensor scheme.C[-1; -2] := mat[1 2; 3 4] * U[3 4; -2] * conj(U[1 2; -1])
    @tensor scheme.E[-1 -2; -3] := scheme.E[1 5; 3] * flip(scheme.T, (3, 4); inv = false)[5 4 -2 2] *
        U[3 4; -3] *
        conj(U[1 2; -1])

    normalize!(scheme.C)
    normalize!(scheme.E)
    return S
end

function lnz(scheme::c4vCTM)
    T_unflipped = permute(flip(scheme.T, (3, 4); inv = false), ((4, 3), (1, 2)))

    corners = fill(scheme.C, 4)
    edges = fill(scheme.E, 4)

    edges[3] = flip(scheme.E, 2; inv = true)
    edges[4] = flip(scheme.E, 2; inv = true)

    return real(log(network_value(T_unflipped, corners, edges)))
end

function build_corner_matrix(scheme)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.C[1; 2] * scheme.E[-1 3; 1] *
        scheme.E[2 4; -3] *
        flip(scheme.T, 3; inv = false)[4 -4 -2 3]
    return mat
end

function find_U_sym(scheme, trunc)
    mat = build_corner_matrix(scheme)
    # avoid symmetry breaking due to numerical accuracy
    mat = 0.5 * (mat + adjoint(mat))

    U, S, _ = svd_trunc(mat; trunc = trunc & trunctol(atol = 1.0e-20))
    return mat, U, S
end

function c4vCTM_init(T::TensorMap{A, S, 0, 4}) where {A, S}
    S_type = scalartype(T)
    Vp = space(T)[1]'
    C = ones(S_type, oneunit(Vp) ← oneunit(Vp))
    E = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    return C, E
end

function Base.show(io::IO, scheme::c4vCTM)
    println(io, "c4vCTM - C4v symmetric Corner Transfer Matrix")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * E: $(summary(scheme.E))")
    return nothing
end
