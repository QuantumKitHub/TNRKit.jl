"""
    struct CFTTransferMatrix{E, S}

A transfer matrix constructed from two rank-4 TNR tensors. The geometry
is labeled by a shape `[h, L, x]` (height, circumference, horizontal shift).

The struct is callable: `(tm::CFTTransferMatrix)(x)` applies the transfer
matrix as a linear map to the input tensor `x`.

!!! note
    `TA` and `TB` are **not** necessarily the original tensors in the network.
    For shapes that require a renormalization step (e.g. `[1, 8, 1]` or
    `[4/√10, 2√10, 2/√10]`), the constructor performs one step of MPO
    disentangling and the stored tensors are the renormalized ones.  The
    `shape` field always labels the *tube geometry*, which determines the
    contraction pattern and modular parameter; it does not change after
    renormalization — the same shape dispatches to the correct reduced action
    (e.g. `[1, 8, 1]` uses the `[1, 4, 0]`-style 1×4 contraction under the
    hood).

# Fields
- `TA::TensorMap{E, S, 2, 2}`: First tensor in the transfer matrix (may be
    renormalized; see note above).
- `TB::TensorMap{E, S, 2, 2}`: Second tensor in the transfer matrix.
    For networks with a one-site unit cell this is equal to `TA`.
- `shape::Vector{Float64}`: Geometry shape triplet `[h, L, x]`.  This labels
    the tube geometry and determines the contraction pattern, domain space,
    and modular parameter formula.
"""
struct CFTTransferMatrix{E, S}
    TA::TensorMap{E, S, 2, 2}
    TB::TensorMap{E, S, 2, 2}
    shape::Vector{Float64}
end

# convenience constructor that converts shape element types
function CFTTransferMatrix(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, shape::Vector{<:Number};
        trunc::TruncationStrategy = notrunc(), truncentanglement::TruncationStrategy = notrunc()
    ) where {E, S}
    if shape ≈ [1, 8, 1] || shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
        # do one step of renormalization
        dl, ur, ul, dr = MPO_opt(TA, TB, trunc, truncentanglement)
        T = reduced_MPO(dl, ur, ul, dr, trunc)
        return CFTTransferMatrix{E, S}(T, T, Float64[shape...])
    else
        return CFTTransferMatrix{E, S}(TA, TB, Float64[shape...])
    end
end

TensorKit.sectortype(::CFTTransferMatrix{E, S}) where {E, S} = sectortype(S)

# ===========================================================================
#  Transfer matrix action as a linear map
# ===========================================================================

# Make the struct callable: (tm)(x) applies the transfer matrix
function (tm::CFTTransferMatrix)(x)
    return _TMaction(tm, x)
end

# Dispatch the action on the shape
function _TMaction(tm::CFTTransferMatrix{E, S}, x) where {E, S}
    if tm.shape ≈ [2, 2, 0]
        return _TMaction_2x2_NtoS(tm, x)
    elseif tm.shape ≈ [sqrt(2) / 2, sqrt(2), sqrt(2) / 2]
        return _TMaction_2x2_NEtoSW(tm, x)
    elseif tm.shape ≈ [1, 4, 0]
        return _TMaction_1x4(tm.TA, tm.TB, x)
    elseif tm.shape ≈ [1, 4, 1]
        return _TMaction_1x4_twist(tm.TA, tm.TB, x)
    elseif tm.shape ≈ [sqrt(2), 2 * sqrt(2), 0]
        return _TMaction_2gates(tm.TA, tm.TB, x)
    elseif tm.shape ≈ [1, 8, 1]
        return _TMaction_1x4(tm.TA, tm.TB, x)
    elseif tm.shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
        return _TMaction_2gates(tm.TA, tm.TB, x)
    else
        error("Unsupported transfer matrix shape: $(tm.shape).")
    end
end

"""
Action of [2, 2, 0] transfer matrix.
"""
function _TMaction_2x2_NtoS(tm::CFTTransferMatrix{E, S}, x) where {E, S}
    @tensor begin
        fx[-1 -2; 5] := twist(tm.TA, 1)[c -1; 1 m] * x[1 2; 5] * tm.TB[m -2; 2 c]
        fx[-1 -2; 5] := twist(tm.TB, 1)[c -1; 1 m] * fx[1 2; 5] * tm.TA[m -2; 2 c]
    end
    return fx
end

"""
Action of [√2/2, √2, √2/2] transfer matrix.
"""
function _TMaction_2x2_NEtoSW(tm::CFTTransferMatrix{E, S}, x) where {E, S}
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
    @tensor begin
        fx[-1 -2 -3 -4; 5] := tm.TB[-2 -3; a b] * x[-1 a b -4; 5]
        fx[-1 -2 -3 -4; 5] := tm.TA[-1 -2; a b] * fx[a b -3 -4; 5]
        fx[-1 -2 -3 -4; 5] := twist(tm.TA, 2)[-3 -4; a b] * fx[-1 -2 a b; 5]
        fx[-1 -2 -3 -4; 5] := twist(tm.TB, 2)[-4 -1; a b] * fx[-3 a b -2; 5]
    end
    return fx
end

"""
Action of [√2, 2√2, 0] transfer matrix, shown in Fig. 25 of https://arxiv.org/pdf/2311.18785. 
First appeared in Chenfeng Bao's thesis: http://hdl.handle.net/10012/14674.
"""
function _TMaction_2gates(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}
    ) where {E, S}
    @tensor begin
        fx[-1 -2 -3 -4; 5] := TB[-1 -2; 1 2] * x[1 2 3 4; 5] * TB[-3 -4; 3 4]
        fx[-1 -2 -3 -4; 5] := TA[-3 -4; 2 3] * fx[1 2 3 4; 5] * TA[-1 -2; 4 1]
    end
    return permute(fx, ((2, 3, 4, 1), (5,)))
end

"""
Action of [1, 4, 0] transfer matrix.
"""
function _TMaction_1x4(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}
    ) where {E, S}
    return @tensor TTTTx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] *
        TA[41 -1; 1 12] * TB[12 -2; 2 23] * TA[23 -3; 3 34] * TB[34 -4; 4 41]
end

"""
Action of [1, 4, 1] transfer matrix.
"""
function _TMaction_1x4_twist(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}
    ) where {E, S}
    TTTTx = _TMaction_1x4(TA, TB, x)
    return permute(TTTTx, ((2, 3, 4, 1), (5,)))
end

# Utility to renormalize tensors for [1, 8, 1] and [4/√10, 2√10, 2/√10]

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

# ===========================================================================
#  Domain space
# ===========================================================================
function TensorKit.domain(tm::CFTTransferMatrix{E, S}) where {E, S}
    if tm.shape ≈ [2, 2, 0]
        return domain(tm.TA, 1) ⊗ domain(tm.TB, 1)
    elseif tm.shape ≈ [sqrt(2) / 2, sqrt(2), sqrt(2) / 2]
        return domain(tm.TA, 1) ⊗ domain(tm.TB, 1) ⊗ domain(tm.TB, 2) ⊗ domain(tm.TA, 2)
    elseif any(isapprox(tm.shape, shape) for shape in [[1, 4, 0], [1, 4, 1], [1, 8, 1]])
        return domain(tm.TA)[1] ⊗ domain(tm.TB)[1] ⊗ domain(tm.TA)[1] ⊗ domain(tm.TB)[1]
    elseif any(isapprox(tm.shape, shape) for shape in [[sqrt(2), 2 * sqrt(2), 0], [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]])
        return domain(tm.TB) ⊗ domain(tm.TB)
    else
        error("Unsupported transfer matrix shape: $(tm.shape).")
    end
end

# ===========================================================================
#  Modular parameter of the transfer matrix
# ===========================================================================
"""
    modular_parameter(tm::CFTTransferMatrix, τ0::Number)

Return the modular parameter `τ_TM` of the transfer matrix, given the
elementary modular parameter `τ0` of the original tensor(s).
"""
function modular_parameter(tm::CFTTransferMatrix, τ0::Number)
    if tm.shape ≈ [2, 2, 0]
        return τ0
    elseif tm.shape ≈ [sqrt(2) / 2, sqrt(2), sqrt(2) / 2]
        return 1 / (1 - τ0)
    elseif tm.shape ≈ [1, 4, 0]
        return τ0 / 4
    elseif tm.shape ≈ [1, 4, 1]
        return (1 + τ0) / 4
    elseif tm.shape ≈ [sqrt(2), 2 * sqrt(2), 0]
        return (1 + τ0) / 2 / (1 - τ0)
    elseif tm.shape ≈ [1, 8, 1]
        τ0′ = (τ0 - 1) / 2
        return τ0′ / 4
    elseif tm.shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
        τ0′ = (τ0 - 1) / 2
        return (1 + τ0′) / 2 / (1 - τ0′)
    else
        error("Unsupported transfer matrix shape: $(tm.shape).")
    end
end

# ===========================================================================
#  Eigenvalue extraction
# ===========================================================================

"""
    leading_eigenvalue(tm::CFTTransferMatrix; Nh = 25)

Compute the leading `Nh` eigenvalues of the transfer matrix in each charge sector.
Returns a `StructuredVector` mapping each fused charge sector to its eigenvalues.

    leading_eigenvalue(tm::CFTTransferMatrix, charge; Nh = 1)

Compute the leading `Nh` eigenvalues in a single charge sector.
Returns a vector of eigenvalues.
"""
function leading_eigenvalue(tm::CFTTransferMatrix{E, S}; Nh::Int = 25) where {E, S}
    @assert Nh >= 1
    I = sectortype(tm.TA)
    if BraidingStyle(I) != Bosonic()
        throw(ArgumentError("Sectors with non-Bosonic charge $I has not been implemented"))
    end

    xspace = domain(tm)
    data = ComplexF64[]
    structure = Dict{I, Vector{Int}}()
    last_index = 1

    for charge in sectors(fuse(xspace))
        vals = leading_eigenvalue(tm, charge; Nh = Nh)
        if length(vals) > 0
            push!(data, vals...)
            structure[charge] = collect(last_index:(last_index + length(vals) - 1))
            last_index += length(vals)
        end
    end

    return StructuredVector(data, structure)
end

function leading_eigenvalue(tm::CFTTransferMatrix{E, S}, charge; Nh::Int = 1) where {E, S}
    @assert Nh >= 1
    I = sectortype(tm.TA)
    V = (I == Trivial) ? field(tm.TA)^1 : Vect[I](charge => 1)
    x = ones(domain(tm) ← V)
    if dim(x) == 0
        error("$charge is not allowed by the transfer matrix.")
    end
    spec, _, info = eigsolve(
        tm, x, Nh, :LM; krylovdim = 40, maxiter = 100, tol = 1.0e-12, verbosity = 0
    )
    if info.converged == 0
        @warn "CFTTransferMatrix eigensolver did not converge in sector $charge."
    end
    return filter(x -> abs(real(x)) ≥ 1.0e-12, spec)
end
