"""
    struct CFTData{E, K, V, A <: AbstractVector{E}}

A struct to hold conformal data extracted from a TNR scheme.

# Constructors
    CFTData(scheme::TNRScheme; kwargs...)
    CFTData(TA::TensorMap{E, S, 2, 2}; kwargs...)
    CFTData(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; kwargs...)

# Fields
    - `central_charge::E`: The central charge of the CFT.
    - `modular_parameter::E`: The elementary modular parameter of a single tensor.
    - `scaling_dimensions::StructuredVector{E, K, V, A}`: The scaling dimensions of the CFT, organized in a `StructuredVector` where the sectors correspond to different spin sectors (or other quantum numbers) and the data contains the scaling dimensions within those sectors

"""
struct CFTData{E, K, V, A <: AbstractVector{E}}
    "Central charge of the CFT."
    central_charge::E
    "Elementary modular parameter for one tensor"
    modular_parameter::E
    "Scaling dimensions of the CFT."
    scaling_dimensions::StructuredVector{E, K, V, A}
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
function CFTData(T::TensorMap{E, S, 2, 2}; shape = [sqrt(2), 2 * sqrt(2), 0], kwargs...) where {E, S}
    if shape == [1, 1, 0] # trivial implementation
        τ0, c = extract_tau_and_c(T)
        Δs = _scaling_dimensions(T, τ0)
        return CFTData(complex(c), τ0, StructuredVector(Δs, Dict([Trivial => collect(eachindex(Δs))])))
    else
        CFTData(T, T; shape, kwargs...)
    end
end

# Main implementation, two-site unitcell
function CFTData(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; shape = [sqrt(2), 2 * sqrt(2), 0],
        trunc = truncrank(16), truncentanglement = trunctol(; rtol = 1.0e-14)
    ) where {E, S}
    norm_const = area_term(TA, TB)^(1 / 4) # canonical normalisation constant
    TA′, TB′ = TA / norm_const, TB / norm_const
    τ0, = extract_tau_and_c(TA′, TB′)
    if shape == [1, 1, 0]
        throw(ArgumentError("The shape [1, 1, 0] is not compatible with a two-site unit cell."))
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
    indices = [[i, -i, -(i + unitcell), i + 1] for i in 1:unitcell]
    indices[end][4] = 1

    T = ncon(fill(T, unitcell), indices)

    outinds = Tuple(collect(1:unitcell))
    ininds = Tuple(collect((unitcell + 1):(2unitcell)))

    T = permute(T, (outinds, ininds))

    data = eig_vals(T)
    data = sort(data; by = x -> abs(x), rev = true) # sorting by magnitude
    data = filter(x -> real(x) > 0, data) # filtering out negative real values
    data = filter(x -> abs(x) > 1.0e-12, data) # filtering out small values

    # modular parameter of the constructed transfer matrix
    Imτ = imag(τ0) / unitcell
    return 1 / (2π * Imτ) .* log.(data[1] ./ data)
end

"""
The "canonical" normalization constant for loop-TNR tensors,
which is the eigenvalue with largest norm of the 2 x 2 transfer matrix.
"""
function area_term(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; is_real = true
    ) where {E, S}
    I = sectortype(TA)
    λ = first(leading_eigenvalue(CFTTransferMatrix(TA, TB, [2, 2, 0]), one(I)))
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
    if BraidingStyle(I) != Bosonic()
        throw(ArgumentError("Sectors with non-Bosonic charge $I has not been implemented"))
    end

    tm = CFTTransferMatrix(TA, TB, shape; trunc, truncentanglement)
    τ = modular_parameter(tm, τ0)

    # eigenvalues of the transfer matrix from all charge sectors
    eigs = leading_eigenvalue(tm; Nh)

    # central charge
    norm_const_0 = eigs[one(I)][1]
    area = shape[1] * shape[2]
    central_charge = 6 / pi / (imag(τ) - imag(τ0) * area / 4) * log(norm_const_0)

    # Construct a StructuredVector of scaling dimensions
    data = ComplexF64[]
    structure = Dict{I, Vector{Int}}()
    last_index = 1
    relative_shift = real(τ) / imag(τ)
    for charge in keys(eigs)
        # DeltaS = Δ - i s Re(τ) / Im(τ)
        DeltaS = -1 / (2 * pi * imag(τ)) * log.(eigs[charge] / norm_const_0)
        if !isapprox(relative_shift, 0; atol = 1.0e-6)
            # save `Δ - i s` in `data`
            push!(data, (real.(DeltaS) + imag.(DeltaS) / relative_shift * im)...)
            structure[charge] = [last_index:(last_index + length(DeltaS) - 1)...]
        else
            # not enough precision to resolve conformal spin
            push!(data, real.(DeltaS)...)
            structure[charge] = [last_index:(last_index + length(DeltaS) - 1)...]
        end
        last_index += length(DeltaS)
    end

    sv = StructuredVector(data, structure)
    sv = sort(sv; by = real)
    sv = filter(x -> real(x) ≤ 1.0e16, sv)
    return CFTData(central_charge, τ0, sv)
end

# Modular parameter and central charge
# ====================================
"""
    extract_tau_and_c(T::TensorMap{E, S, 2, 2}) where {E, S}
    extract_tau_and_c(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}) where {E, S}

Compute the modular parameter τ of one tensor and the central charge c.
"""
function extract_tau_and_c(T::TensorMap{E, S, 2, 2}) where {E, S}
    return extract_tau_and_c(T, T)
end
function extract_tau_and_c(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    I = sectortype(TA)
    # shorthand: leading eigenvalue in the identity sector
    _λ(TA, TB, shape) = real(first(leading_eigenvalue(CFTTransferMatrix(TA, TB, shape), one(I))))
    shape1, p1 = [2, 2, 0], ((3, 1), (4, 2))
    shape2, p2 = [sqrt(2) / 2, sqrt(2), sqrt(2) / 2], ((2, 4), (1, 3))
    # N → S
    λv = _λ(TA, TB, shape1)
    # E → W
    λh = _λ(permute(TB, p1), permute(TA, p1), shape1)
    # NE → SW
    λa = _λ(TA, TB, shape2)
    # NW → SE
    λb = _λ(permute(TB, p2), permute(TA, p2), shape2)
    # edge case: c = 0
    if all(isapprox.(λv, (λh, λa, λb); rtol = 1.0e-6))
        return complex(0.0, 1.0), 0.0
    end
    # when c ≠ 0
    a1, a2, a3 = log(λh / λv), log(λa / λv), log(λb / λv)
    # c here is actually π c / 6
    c, v, θ = solve_cvtheta(a1, a2, a3)
    return v * cis(θ), 6 * c / pi
end

# Logistic sigmoid and its inverse (logit)
sigmoid(x) = 1 / (1 + exp(-x))
logit(p) = log(p / (1 - p))

"""
    solve_cvtheta(a1, a2, a3; c0 = 0.5, v0 = 1.0, θ0 = π / 2)

Solve for positive (c, v) and θ ∈ (0, π) from the three equations:

    a1 = (1/v - v) * c * sin(θ)
    a2 = (v / (1 + v² - 2v cos θ) - v) * c * sin(θ)
    a3 = (v / (1 + v² + 2v cos θ) - v) * c * sin(θ)

Returns `(c, v, θ)`.
"""
function solve_cvtheta(a1, a2, a3; c0 = 0.5, v0 = 1.0, θ0 = π / 2)
    function f!(du, u, p)
        xc, xv, xθ = u
        # Work in unconstrained coords to keep variables in their natural domain
        c = exp(xc)         # make c > 0
        v = exp(xv)         # make v > 0
        θ = π * sigmoid(xθ) # make θ ∈ (0, π)

        sinθ = sin(θ)
        cosθ = cos(θ)

        du[1] = (1 / v - v) * c * sinθ - a1
        du[2] = (v / (1 + v^2 - 2v * cosθ) - v) * c * sinθ - a2
        du[3] = (v / (1 + v^2 + 2v * cosθ) - v) * c * sinθ - a3
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
    c = exp(xc)
    v = exp(xv)
    θ = π * sigmoid(xθ)
    return c, v, θ
end
