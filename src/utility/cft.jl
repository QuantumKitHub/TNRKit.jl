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
    elseif (shape ≈ [sqrt(2), 2 * sqrt(2), 0]) || (shape == [1, 4, 1]) # these shapes need no truncation
        return spec(TA′, TB′, shape, τ0)
    elseif (shape == [1, 8, 1]) || (shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)])
        dl, ur, ul, dr = MPO_opt(TA′, TB′, trunc, truncentanglement)
        T = reduced_MPO(dl, ur, ul, dr, trunc)
        return spec(T, T, shape, τ0)
    else
        throw(ArgumentError("Shape $shape is not implemented."))
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
    f0(x) = @tensor begin
        # for fermions, use APBC (NS) sector
        fx[-1 -2] := twist(TA, 1)[c -1; 1 m] * x[1 2] * TB[m -2; 2 c]
        fx[-1 -2] := twist(TB, 1)[c -1; 1 m] * fx[1 2] * TA[m -2; 2 c]
    end
    x0 = ones(domain(TA, 1) ⊗ domain(TB, 1))
    spec0, _, info = eigsolve(f0, x0, 1, :LM; verbosity = 0)
    if info.converged == 0
        @warn "The area term eigensolver did not converge."
    end
    if is_real
        return real(spec0[1])
    else
        return spec0[1]
    end
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
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2},
        shape::Vector{<:Number}, τ0::Number; Nh = 25
    ) where {E, S}
    I = sectortype(TA)
    𝔽 = field(TA)
    if BraidingStyle(I) != Bosonic()
        throw(ArgumentError("Sectors with non-Bosonic charge $I has not been implemented"))
    end

    # eigenvalues of the transfer matrix
    xspace, f, τ = if shape ≈ [1, 4, 1]
        domain(TA)[1] ⊗ domain(TB)[1] ⊗ domain(TA)[1] ⊗ domain(TB)[1],
            MPO_action_1x4_twist, (1 + τ0) / 4
    elseif shape ≈ [sqrt(2), 2 * sqrt(2), 0]
        domain(TB) ⊗ domain(TB), MPO_action_2gates, (1 + τ0) / 2 / (1 - τ0)
        # in the following cases, (TA, TB) are no longer the original tensor in the network
    elseif shape ≈ [1, 8, 1]
        τ0′ = (τ0 - 1) / 2
        domain(TA)[1] ⊗ domain(TB)[1] ⊗ domain(TA)[1] ⊗ domain(TB)[1],
            MPO_action_1x4, τ0′ / 4
    elseif shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
        τ0′ = (τ0 - 1) / 2
        domain(TB) ⊗ domain(TB), MPO_action_2gates, (1 + τ0′) / 2 / (1 - τ0′)
    else
        error("Unsupported transfer matrix shape.")
    end
    spec_sector = Dict(
        map(sectors(fuse(xspace))) do charge
            V = (I == Trivial) ? 𝔽^1 : Vect[I](charge => 1)
            x = ones(xspace ← V)
            if dim(x) == 0
                return charge => [0.0]
            else
                spec, _, info = eigsolve(
                    a -> f(TA, TB, a), x, Nh, :LM; krylovdim = 40, maxiter = 100,
                    tol = 1.0e-12,
                    verbosity = 0
                )
                if info.converged == 0
                    @warn "The spectrum eigensolver in sector $charge did not converge."
                end
                return charge => filter(x -> abs(real(x)) ≥ 1.0e-12, spec)
            end
        end
    )

    # central charge
    norm_const_0 = spec_sector[one(I)][1]
    area = shape[1] * shape[2]
    central_charge = 6 / pi / (imag(τ) - imag(τ0) * area / 4) * log(norm_const_0)

    # Construct a StructuredVector from the data of the different sectors
    data = ComplexF64[]
    structure = Dict{eltype(sectors(fuse(xspace))), Vector{Int}}()
    last_index = 1
    relative_shift = real(τ) / imag(τ)
    for charge in sectors(fuse(xspace))
        # DeltaS = Δ - i s Re(τ) / Im(τ)
        DeltaS = -1 / (2 * pi * imag(τ)) * log.(spec_sector[charge] / norm_const_0)
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

function MPO_opt(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2},
        trunc::TruncationStrategy, truncentanglement::TruncationStrategy
    ) where {E, S}
    pretrunc = truncrank(2 * trunc.howmany)
    dl, ur = SVD12(TA, pretrunc)
    dr, ul = SVD12(transpose(TB, ((2, 4), (1, 3))), pretrunc)

    transfer_MPO = [
        transpose(dl, ((1,), (3, 2))), ur, transpose(ul, ((2,), (3, 1))),
        transpose(dr, ((3,), (2, 1))),
    ]

    in_inds = [1, 1, 1, 1]
    out_inds = [1, 2, 2, 1]
    MPO_function(steps, data) = abs(data[end])
    criterion = maxiter(10) & convcrit(1.0e-12, MPO_function)
    PR_list, PL_list = find_projectors(
        transfer_MPO, in_inds, out_inds, criterion,
        trunc & truncentanglement
    )

    MPO_disentangled!(transfer_MPO, in_inds, out_inds, PR_list, PL_list)
    return transfer_MPO
end

# Apply functions for diagonalising different shapes of transfer matrices
# =======================================================================
# Fig.25 of https://arxiv.org/pdf/2311.18785. Firstly appear in Chenfeng Bao's thesis, see http://hdl.handle.net/10012/14674.
"""
When the elementary modular parameter for TA, TB is `τ`,
the transfer matrix has `τ_TM = (1 + τ) / 2 / (1 - τ)`.
"""
function MPO_action_2gates(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}
    ) where {E, S}
    @tensor begin
        fx[-1 -2 -3 -4; 5] := TB[-1 -2; 1 2] * x[1 2 3 4; 5] * TB[-3 -4; 3 4]
        fx[-1 -2 -3 -4; 5] := TA[-3 -4; 2 3] * fx[1 2 3 4; 5] * TA[-1 -2; 4 1]
    end
    return permute(fx, ((2, 3, 4, 1), (5,)))
end

"""
When the elementary modular parameter for TA, TB is `τ`,
the transfer matrix has `τ_TM = τ / 4`.
"""
function MPO_action_1x4(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}
    ) where {E, S}
    return @tensor TTTTx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] *
        TA[41 -1; 1 12] * TB[12 -2; 2 23] * TA[23 -3; 3 34] * TB[34 -4; 4 41]
end

"""
When the elementary modular parameter for TA, TB is `τ`,
the transfer matrix has `τ_TM = (τ + 1) / 4`.
"""
function MPO_action_1x4_twist(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}
    ) where {E, S}
    TTTTx = MPO_action_1x4(TA, TB, x)
    return permute(TTTTx, ((2, 3, 4, 1), (5,)))
end

"""
This renormalization will change the elementary
modular parameter from τ to (τ - 1) / 2.
"""
function reduced_MPO(
        dl::TensorMap{E, S, 1, 2}, ur::TensorMap{E, S, 1, 2},
        ul::TensorMap{E, S, 1, 2}, dr::TensorMap{E, S, 1, 2},
        trunc::TruncationStrategy
    ) where {E, S}
    @plansor temp[-1 -2; -3 -4] :=
        ur[-1; 1 4] * ul[4; 3 -2] * dr[-3; 2 1] * dl[2; -4 3]
    D, U = SVD12(temp, trunc)
    @plansor translate[-1 -2; -3 -4] := U[-2; 1 -4] * D[-1 1; -3]
    return translate
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
    # leading eigenvalue of each transfer matrix
    # corresponding to the Δ = s = 0 identity field
    λv = real(_eigsolve_2x2_NtoS(TA, TB))
    λh = real(_eigsolve_2x2_EtoW(TA, TB))
    λa = real(_eigsolve_2x2_NEtoSW(TA, TB))
    λb = real(_eigsolve_2x2_NWtoSE(TA, TB))
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

function _eigsolve_2x2_NtoS(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    f0(x) = @tensor begin
        # for fermions, use APBC (NS) sector
        fx[-1 -2] := twist(TA, 1)[c -1; 1 m] * x[1 2] * TB[m -2; 2 c]
        fx[-1 -2] := twist(TB, 1)[c -1; 1 m] * fx[1 2] * TA[m -2; 2 c]
    end
    x0 = ones(domain(TA, 1) ⊗ domain(TB, 1))
    spec0, _, info = eigsolve(f0, x0, 1, :LM; verbosity = 0)
    if info.converged == 0
        @warn "The area term eigensolver did not converge."
    end
    return first(spec0)
end

function _eigsolve_2x2_EtoW(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    TA′ = permute(TA, ((3, 1), (4, 2)))
    TB′ = permute(TB, ((3, 1), (4, 2)))
    return _eigsolve_2x2_NtoS(TB′, TA′)
end

function _eigsolve_2x2_NEtoSW(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    #= 
        1   2
        ┌---┬---┐
        |   |   |
    3'--A---B---┤ 3
        |   |   |
    4'--B---A---┘ 4
        |   |
        1'  2'
    =#
    f0(x) = @tensor begin
        # for fermions, use APBC (NS) sector
        fx[-1 -2 -3 -4] := TB[-2 -3; a b] * x[-1 a b -4]
        fx[-1 -2 -3 -4] := TA[-1 -2; a b] * fx[a b -3 -4]
        fx[-1 -2 -3 -4] := twist(TA, 2)[-3 -4; a b] * fx[-1 -2 a b]
        fx[-1 -2 -3 -4] := twist(TB, 2)[-4 -1; a b] * fx[-3 a b -2]
    end
    x0 = ones(domain(TA, 1) ⊗ domain(TB, 1) ⊗ domain(TB, 2) ⊗ domain(TA, 2))
    spec0, _, info = eigsolve(f0, x0, 1, :LM; verbosity = 0)
    if info.converged == 0
        @warn "The area term eigensolver did not converge."
    end
    return first(spec0)
end

function _eigsolve_2x2_NWtoSE(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    TA′ = permute(TA, ((2, 4), (1, 3)))
    TB′ = permute(TB, ((2, 4), (1, 3)))
    return _eigsolve_2x2_NEtoSW(TB′, TA′)
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
