"""
    struct CFTData{E, K, A <: AbstractVector{E}}

A struct to hold conformal data extracted from a TNR scheme.

# Constructors
    CFTData(scheme::TNRScheme; kwargs...)
    CFTData(TA::TensorMap{E, S, 2, 2}; kwargs...)
    CFTData(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; kwargs...)

# Fields
    - `central_charge::E`: The central charge of the CFT.
    - `modular_parameter::E`: The elementary modular parameter of a single tensor.
    - `scaling_dimensions::StructuredVector{E, K, A}`: The scaling dimensions of the CFT, organized in a `StructuredVector` where the sectors correspond to different spin sectors (or other quantum numbers) and the data contains the scaling dimensions within those sectors

"""
struct CFTData{E, K, A <: AbstractVector{E}}
    "Central charge of the CFT."
    central_charge::E
    "Elementary modular parameter for one tensor"
    modular_parameter::E
    "Scaling dimensions of the CFT."
    scaling_dimensions::StructuredVector{E, K, A}
end

function Base.show(io::IO, data::CFTData)
    println(io, "CFTData")
    println(io, "  * central charge: $(data.central_charge)")
    println(io, "  * scaling dimensions: $(data.scaling_dimensions)")
    return nothing
end

CFTData(scheme::TNRScheme; kwargs...) = CFTData(scheme.T; kwargs...) # simple 1-site unitcell schemes
CFTData(scheme::LoopTNR; kwargs...) = CFTData(scheme.TA, scheme.TB; kwargs...) # 2-site unitcell schemes
function CFTData(scheme::BTRG; kwargs...) # merge bond tensors into central tensor
    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    return CFTData(T_unit; kwargs...)
end

# one-site unitcell
function CFTData(
        T::TensorMap{E, S, 2, 2}; shape = [sqrt(2), 2 * sqrt(2), 0], fast_tau_alg::Bool = true, kwargs...
    ) where {E, S}
    if shape == [1, 1, 0] # trivial implementation
        τ0, c = extract_tau_and_c(T; fast = fast_tau_alg)
        Δs = _scaling_dimensions(T, τ0)
        return CFTData(complex(c), τ0, Δs)
    else
        CFTData(T, T; shape, fast_tau_alg, kwargs...)
    end
end

# Main implementation, two-site unitcell
function CFTData(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2};
        shape = [sqrt(2), 2 * sqrt(2), 0], fast_tau_alg::Bool = true,
        trunc = truncrank(16), truncentanglement = trunctol(; rtol = 1.0e-14)
    ) where {E, S}
    norm_const = area_term(TA, TB)^(1 / 4) # canonical normalisation constant
    TA′, TB′ = TA / norm_const, TB / norm_const
    τ0, = extract_tau_and_c(TA′, TB′; fast = fast_tau_alg)
    if shape[1] ≈ 1 && shape[2] != 0 && shape[3] == 0
        throw(ArgumentError("The shape [1, L, 0] is not compatible with a two-site unit cell."))
    else
        return spec(TA′, TB′, shape, τ0; trunc, truncentanglement)
    end
end

"""
Construct the transfer matrix along vertical direction
with `unitcell` copies of `T` concatenated horizontally.
`τ0` is the modular parameter of a single `T`.
"""
function _scaling_dimensions(T::TensorMap{E, S, 2, 2}, τ0::Number; unitcell = 1) where {E, S}
    sv = _rowtm_eigvals(T, unitcell)
    sv = filter(x -> real(x) > 0 && abs(x) > 1.0e-12, sv)
    isempty(sv) && throw(ArgumentError("No valid eigenvalues found in transfer matrix spectrum."))

    λ0 = argmax(abs, sv.data)
    Imτ = imag(τ0) / unitcell
    Δs = 1 / (2π * Imτ) .* log.(λ0 ./ sv)
    return sort(Δs; by = real)
end

"""
The "canonical" normalization constant for loop-TNR tensors,
which is the eigenvalue with largest norm of the 2 x 2 transfer matrix.
"""
function area_term(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; is_real = true
    ) where {E, S}
    I = sectortype(TA)
    pbc = (BraidingStyle(I) == Fermionic()) ? false : true
    λ = first(leading_eigenvalue(CFTTransferMatrix(TA, TB, [2, 2, 0]), one(I); pbc, Nh = 1))
    return is_real ? real(λ) : λ
end

# The case with spin is based on https://arxiv.org/pdf/1512.03846 and some private communications with Yingjie Wei and Atsushi Ueda
"""
Internal function to construct transfer matrices and extract conformal data.

# Arguments
- `TA, TB`: Rank-4 network tensors (may be identical for 1-site unit cells).
- `shape`:  A triplet `[h, L, x]` — height, circumference, and horizontal shift
  of the tube geometry, in units of the original tensor patch.
- `τ0`:     Elementary modular parameter of one tensor.
- `Nh`:     Number of eigenvalues to solve for per sector (default 25).
"""
function spec(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, shape::Vector{<:Number},
        τ0::Number; Nh = 25, trunc = notrunc(), truncentanglement = notrunc()
    ) where {E, S}
    I = sectortype(TA)
    tm = CFTTransferMatrix(TA, TB, shape; trunc, truncentanglement)
    τ = modular_parameter(tm, τ0)

    # eigenvalues of the transfer matrix from all charge sectors
    eigs = leading_eigenvalue(tm; Nh)

    # central charge
    λ0 = if BraidingStyle(I) == Fermionic()
        eigs[(:NS, one(I))][1]
    else
        eigs[one(I)][1]
    end
    area = shape[1] * shape[2]
    central_charge = 6 / pi / (imag(τ) - imag(τ0) * area / 4) * log(λ0)

    # scaling dimension and conformal spin
    # DeltaS = Δ - i s Re(τ) / Im(τ)
    Reτ, Imτ = real(τ), imag(τ)
    relative_shift = Reτ / Imτ
    DeltaS = -1 / (2 * pi * Imτ) * log.(eigs / λ0)
    if !isapprox(relative_shift, 0; atol = 1.0e-6)
        sv = real.(DeltaS) + imag.(DeltaS) / relative_shift * im
    else
        # not enough precision to resolve conformal spin
        sv = complex.(real.(DeltaS))
    end
    sv = sort(sv; by = real)
    sv = filter(x -> real(x) ≤ 1.0e16, sv)
    return CFTData(central_charge, τ0, sv)
end

# Modular parameter and central charge
# ====================================

# Utility functions
sigmoid(x) = 1 / (1 + exp(-x))
logit(p) = log(p / (1 - p))
function _find_λ0(TA, TB, shape)
    I = sectortype(TA)
    charge = one(I)
    pbc = (BraidingStyle(I) == Fermionic()) ? false : true
    λs = leading_eigenvalue(CFTTransferMatrix(TA, TB, shape), charge; pbc, Nh = 1)
    return real(first(λs))
end

"""
    extract_tau_and_c(T::TensorMap{E, S, 2, 2}; fast::Bool = true) where {E, S}
    extract_tau_and_c(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; fast::Bool = true) where {E, S}

Compute the modular parameter τ of one tensor and the central charge c.
When `fast = true`, 1x2 transfer matrices are used, which runs faster but
produced slightly less accurate result. Otherwise, 2x2 transfer matrices are used.
"""
function extract_tau_and_c(T::TensorMap{E, S, 2, 2}; kwargs...) where {E, S}
    return extract_tau_and_c(T, T; kwargs...)
end
function extract_tau_and_c(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; fast::Bool = true
    ) where {E, S}
    return fast ? _extract_tau_and_c_1x2(TA, TB) : _extract_tau_and_c_2x2(TA, TB)
end

function _extract_tau_and_c_1x2(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    shape1, p1 = [1, 2, 1], ((3, 1), (4, 2))
    shape2, p2 = [sqrt(2), sqrt(2), 0], ((4, 2), (3, 1))
    # N → S (1x2): τ1 = (1 + τ) / 2
    λv = _find_λ0(TA, TB, shape1)
    # E → W (1x2): τ2 = (τ - 1) / (2 τ)
    λh = _find_λ0(permute(TB, p1), permute(TA, p1), shape1)
    # NE → SW (2x1): τ3 = (1 + τ) / (1 - τ)
    λa = _find_λ0(TA, TB, shape2)
    # NW → SE (2x1): τ4 = (τ - 1) / (τ + 1)
    λb = _find_λ0(permute(TB, p2), permute(TA, p2), shape2)
    # edge case: c = 0
    if all(isapprox.(λv, (λh, λa, λb); rtol = 1.0e-6))
        return complex(0.0, 1.0), 0.0
    end
    # when c ≠ 0
    a1, a2, a3 = log(λh / λv), log(λa / λv), log(λb / λv)
    # c here is actually π c / 6
    c, v, θ = solve_cvtheta(a1, a2, a3; fast = true)
    return v * cis(θ), 6 * c / pi
end

function _extract_tau_and_c_2x2(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    shape1, p1 = [2, 2, 0], ((3, 1), (4, 2))
    # N → S: τ1 = τ
    λv = _find_λ0(TA, TB, shape1)
    # E → W: τ2 = -1 / τ
    λh = _find_λ0(permute(TB, p1), permute(TA, p1), shape1)

    shape2, p2 = [sqrt(2) / 2, sqrt(2), sqrt(2) / 2], ((2, 4), (1, 3))
    # NE → SW: τ3 = 1 / (1 - τ)
    λa = _find_λ0(TA, TB, shape2)
    # NW → SE: τ4 = τ / (1 + τ)
    λb = _find_λ0(permute(TB, p2), permute(TA, p2), shape2)

    # edge case: c = 0
    if all(isapprox.(λv, (λh, λa, λb); rtol = 1.0e-6))
        return complex(0.0, 1.0), 0.0
    end
    # when c ≠ 0
    a1, a2, a3 = log(λh / λv), log(λa / λv), log(λb / λv)
    # c here is actually π c / 6
    c, v, θ = solve_cvtheta(a1, a2, a3; fast = false)
    return v * cis(θ), 6 * c / pi
end

"""
    solve_cvtheta(a1, a2, a3; c0 = 0.5, v0 = 1.0, θ0 = π / 2, fast::Bool = true)

Solve for positive (c, v) and θ ∈ (0, π).
"""
function solve_cvtheta(
        a1, a2, a3; c0 = 0.5, v0 = 1.0, θ0 = π / 2, fast::Bool = true
    )
    function f!(du, u, p)
        xc, xv, xθ = u
        # Work in unconstrained coords to keep variables in their natural domain
        c = exp(xc)         # make c > 0
        v = exp(xv)         # make v > 0
        θ = π * sigmoid(xθ) # make θ ∈ (0, π)

        sinθ, cosθ = sin(θ), cos(θ)
        vm, vp = 1 + v^2 - 2v * cosθ, 1 + v^2 + 2v * cosθ
        if fast
            csinθ = c * sinθ / 2
            du[1] = (1 / v - v) * csinθ - a1
            du[2] = (4 / vm - 1) * v * csinθ - a2
            du[3] = (4 / vp - 1) * v * csinθ - a3
        else
            csinθ = c * sinθ
            du[1] = (1 / v - v) * csinθ - a1
            du[2] = (1 / vm - 1) * v * csinθ - a2
            du[3] = (1 / vp - 1) * v * csinθ - a3
        end
        return nothing
    end

    # Initial guess in unconstrained space
    u0 = [log(c0), log(v0), logit(θ0 / π)]
    prob = NonlinearProblem(f!, u0)
    sol = solve(prob, NewtonRaphson(; autodiff = AutoForwardDiff()))
    if !SciMLBase.successful_retcode(sol)
        @warn "Solver did not converge" retcode = sol.retcode resid = sol.resid
    end
    xc, xv, xθ = sol.u
    return exp(xc), exp(xv), π * sigmoid(xθ)
end
