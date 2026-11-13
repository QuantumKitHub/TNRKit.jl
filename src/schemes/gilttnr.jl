"""
$(TYPEDEF)

Graph-Independent Local Truncations + Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T [; ε=1e-5, maxiter=20])

# Running the algorithm
    run!(::GILTTNR, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step first applies Gilt to the four legs of a plaquette of a checkerboard lattice,
followed by two TRG steps. Each step thus rescales the lattice by a (linear) factor of 2 (use `scalefactor = 4.0` in [`free_energy`](@ref))
and rotates the lattice by 90 degrees.
The elementary modular parameter `τ₀ ↦ -1 / τ₀`.

`ε` is the threshold below which (normalized) environment singular values are truncated by Gilt,
it should be lowered as the bond dimension is increased.
In each step, Gilt sweeps over the four legs of the plaquette until no leg can be truncated any further,
or until `maxiter` sweeps have been done.

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step
    - 3: Print information about the Gilt sweeps

# Fields

$(TYPEDFIELDS)

# References
* [Hauru et. al. Phys. Rev. B 97 (2018)](@cite hauru2018)
"""
mutable struct GILTTNR{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT

    "Gilt threshold"
    ε::Float64

    "Maximum number of Gilt sweeps per step"
    maxiter::Int

    function GILTTNR(T::TT; ε::Real = 1.0e-5, maxiter::Int = 20) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T, ε, maxiter)
    end
end

# Threshold on the singular values of R′ for a leg to be considered converged
const gilt_convergence_ε = 1.0e-2

# Maximal recursion depth when optimizing R′, guards against R′ spectra that stall just outside `gilt_convergence_ε`
const gilt_max_recursion = 20

# Internal stopping criterion for the Gilt sweeps: stop once no leg of the plaquette can be truncated any further
struct giltcrit <: stopcrit end

(crit::giltcrit)(steps::Int, done_legs) = !all(values(done_legs))

function stopping_info(::giltcrit, steps::Int, done_legs)
    return "Gilt criterion reached: all legs converged after $steps sweeps"
end

#=
The plaquette that is Gilted, in the [W S; N E] leg convention:

        |       |
     --T2--N---T1--
        |       |
        W       E
        |       |
     --T1--S---T2--
        |       |

In the checkerboard lattice every bond borders exactly one such plaquette,
so the four legs N, E, S, W cover all bonds of the lattice.
=#

function step!(scheme::GILTTNR, trunc::TruncationStrategy)
    # step 1: Gilt on the checkerboard lattice
    T1, T2 = gilt_plaquette(scheme.T, scheme.T, scheme.ε, scheme.maxiter)

    # step 2: TRG on the checkerboard lattice, the new tensors live on the plaquettes that were not Gilted
    A, B = SVD12(T2, trunc)
    C, D = SVD12(transpose(T1, ((2, 4), (1, 3))), trunc)
    @plansor T[-1 -2; -3 -4] := D[-2; 1 2] * B[-1; 4 1] * C[4 3; -3] * A[3 2; -4]

    # step 3: TRG on the homogeneous lattice
    A, B = SVD12(T, trunc)
    C, D = SVD12(transpose(T, ((2, 4), (1, 3))), trunc)
    @plansor scheme.T[-1 -2; -3 -4] := D[-2; 1 2] * B[-1; 4 1] * C[4 3; -3] * A[3 2; -4]
    return scheme
end

function gilt_plaquette(T1, T2, ε::Real, max_sweeps::Int)
    crit = giltcrit() & maxiter(max_sweeps)
    done_legs = Dict(leg => false for leg in (:S, :N, :E, :W))
    sweeps = 0
    t = @elapsed while crit(sweeps, done_legs)
        for leg in (:S, :N, :E, :W)
            T1, T2, done_legs[leg] = gilt_leg(T1, T2, leg, ε)
        end
        sweeps += 1
        @infov 3 "Gilt sweep $sweeps: bond dimensions [W S; N E] of T1: $(dim.(space.(Ref(T1), 1:4)))"
    end
    @infov 3 "Gilt finished\n $(stopping_info(crit, sweeps, done_legs))\n Elapsed time: $(t)s"
    return T1, T2
end

"""
Truncate a single leg of the plaquette with Gilt.
Returns the new tensors and whether the leg could not be truncated any further.
"""
function gilt_leg(T1, T2, leg::Symbol, ε::Real)
    U, S = gilt_environment(T1, T2, leg)
    R = gilt_optimize_R(U, S, ε)

    # split R′ = R1 * R2, truncating singular values that do not contribute to R′
    u, s, v = svd_trunc(R; trunc = truncerror(; rtol = ε * 1.0e-3))
    done = all(x -> abs(x - 1) < gilt_convergence_ε, s.data)
    R1, R2 = u * sqrt(s), sqrt(s) * v

    T1, T2 = gilt_absorb(T1, T2, R1, R2, leg)
    return T1, T2, done
end

"""
Environment spectrum of a leg of the plaquette.

Returns the left singular vectors `U` and the singular values `S` (normalized to sum to one)
of the plaquette environment with `leg` cut. `U[x y; w]` has as first leg the side of the bond
that is a codomain leg of its tensor and as second leg the side that is a domain leg of its tensor.
"""
function gilt_environment(T1, T2, leg::Symbol)
    # corners of the plaquette, outer legs traced out
    @tensor begin
        CTL[s e; s′ e′] := T2[w s; n e] * conj(T2[w s′; n e′])
        CTR[w s; w′ s′] := T1[w s; n e] * conj(T1[w′ s′; n e])
        CBL[n e; n′ e′] := T1[w s; n e] * conj(T1[w s; n′ e′])
        CBR[w n; w′ n′] := T2[w s; n e] * conj(T2[w′ s; n′ e])
    end

    if leg === :N
        @tensor opt = true E[-1 -2; -3 -4] := CTR[-1 1; -3 2] * CBR[3 1; 4 2] * CBL[5 3; 6 4] * CTL[5 -2; 6 -4]
    elseif leg === :E
        @tensor opt = true E[-1 -2; -3 -4] := CTR[1 -1; 2 -3] * CTL[3 1; 4 2] * CBL[3 5; 4 6] * CBR[5 -2; 6 -4]
    elseif leg === :S
        @tensor opt = true E[-1 -2; -3 -4] := CBR[-1 1; -3 2] * CTR[3 1; 4 2] * CTL[5 3; 6 4] * CBL[5 -2; 6 -4]
    elseif leg === :W
        @tensor opt = true E[-1 -2; -3 -4] := CTL[-1 1; -3 2] * CTR[1 3; 2 4] * CBR[5 3; 6 4] * CBL[-2 5; -4 6]
    else
        throw(ArgumentError("Invalid leg: $leg"))
    end

    S², U = eigh_full(project_hermitian!(E))
    S = DiagonalTensorMap(sqrt.(abs.(S².data)), space(S², 1))
    S /= tr(S)
    return U, S
end

"""
Construct the matrix R′ that replaces the identity on the cut leg, recursively applying Gilt
until the leg cannot be truncated further. The returned `R′[a; b]` has its codomain leg on the
side of the tensor whose domain leg forms the bond.
"""
function gilt_optimize_R(U, S, ε::Real; depth::Int = 1)
    # t′ minimizes |(t - t′)S|² + ε²|t′|², t is the identity on the bond (closing the loop needs a twist for fermions)
    @tensor t[-1] := twist(U, 1)[1 1; -1]
    weight = S^2 * inv(S^2 + ε^2 * id(domain(S)))
    @tensor t′[-1] := weight[1; -1] * t[1]
    @tensor R[-1; -2] := conj(U[-2 -1; 1]) * t′[1]

    u, s, v = svd_trunc(R; trunc = truncerror(; rtol = ε * 1.0e-3))
    if depth < gilt_max_recursion && !all(x -> abs(x - 1) < gilt_convergence_ε, s.data)
        us, sv = u * sqrt(s), sqrt(s) * v
        @tensor UR[-1 -2; -3] := U[1 2; 3] * sv[-1; 1] * us[2; -2] * S[3; -3]
        Uinner, Sinner, _ = svd_compact(UR)
        Sinner /= tr(Sinner)
        R = us * gilt_optimize_R(Uinner, Sinner, ε; depth = depth + 1) * sv
    end
    return R
end

function gilt_absorb(T1, T2, R1, R2, leg::Symbol)
    if leg === :N
        @tensor T2[-1 -2; -3 -4] := T2[-1 -2; -3 1] * R1[1; -4]
        @tensor T1[-1 -2; -3 -4] := R2[-1; 1] * T1[1 -2; -3 -4]
    elseif leg === :E
        @tensor T2[-1 -2; -3 -4] := T2[-1 -2; 1 -4] * R1[1; -3]
        @tensor T1[-1 -2; -3 -4] := R2[-2; 1] * T1[-1 1; -3 -4]
    elseif leg === :S
        @tensor T1[-1 -2; -3 -4] := T1[-1 -2; -3 1] * R1[1; -4]
        @tensor T2[-1 -2; -3 -4] := R2[-1; 1] * T2[1 -2; -3 -4]
    elseif leg === :W
        @tensor T1[-1 -2; -3 -4] := T1[-1 -2; 1 -4] * R1[1; -3]
        @tensor T2[-1 -2; -3 -4] := R2[-2; 1] * T2[-1 1; -3 -4]
    else
        throw(ArgumentError("Invalid leg: $leg"))
    end
    return T1, T2
end

function Base.show(io::IO, scheme::GILTTNR)
    println(io, "Gilt-TNR - Graph-Independent Local Truncations + TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * ε: $(scheme.ε)")
    println(io, "  * maxiter: $(scheme.maxiter)")
    return nothing
end
