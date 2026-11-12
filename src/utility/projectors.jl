"""
Perform SVD of `t` in the "reversed" direction such that
`t = u * s * vh` and the arrow direction is `u → s → vh`.
"""
function svd_reversed(t::AbstractTensorMap; kwargs...)
    vh, s, u, ϵ = svd_trunc(transpose(t); kwargs...)
    u, s, vh = transpose(u), transpose(s), transpose(vh)
    @assert isdual(space(s, 1))
    return u, DiagonalTensorMap(s), vh, ϵ
end

#Utility functions for QR decomp
function QR_L(
        L::TT, T::AbstractTensorMap{E, S, M, N},
        in_ind::Int, out_ind::Int
    )::TT where {E, S, TT <: AbstractTensorMap{E, S, 1, 1}, M, N}
    permT = (
        (in_ind,),
        (
            reverse(collect(1:(in_ind - 1)))..., collect((M + 1):(M + N))...,
            reverse(collect((in_ind + 1):M))...,
        ),
    )
    permLT = (
        (
            reverse(collect(2:(in_ind + out_ind - 1)))..., 1,
            reverse(collect((in_ind + out_ind + 1):(M + N)))...,
        ), (in_ind + out_ind,),
    )
    LT = transpose(L * transpose(T, permT), permLT)
    _, Rt = left_orth(LT)
    return normalize!(Rt, Inf)
end

function QR_R(
        R::TT, T::AbstractTensorMap{E, S, M, N},
        in_ind::Int, out_ind::Int
    )::TT where {E, S, TT <: AbstractTensorMap{E, S, 1, 1}, M, N}
    permT = (
        (
            reverse(collect((M + 1):(M + in_ind - 1)))..., collect(1:M)...,
            reverse(collect((M + in_ind + 1):(M + N)))...,
        ), (M + in_ind,),
    )
    permTR = (
        (in_ind + out_ind - 1,),
        (
            reverse(collect(1:(in_ind + out_ind - 2)))..., M + N,
            reverse(collect((in_ind + out_ind):(M + N - 1)))...,
        ),
    )
    TR = transpose(transpose(T, permT) * R, permTR)
    Lt, _ = right_orth(TR)
    return normalize!(Lt, Inf)
end

# Functions to find the left and right projectors

# Function to find the list of left projectors L_list
function find_L(
        psi::Vector{T}, in_inds::Vector{Int}, out_inds::Vector{Int},
        entanglement_criterion::stopcrit
    ) where {T <: AbstractTensorMap}
    type = eltype(psi[1])
    n = length(psi)
    L_list = map(1:n) do i
        L = id(type, codomain(psi[i])[in_inds[i]])
        error = [Inf]
        crit = true
        steps = 1
        while crit
            L_last_time = L
            for j in 0:(n - 1)
                running_pos = mod(i + j - 1, n) + 1
                L = QR_L(
                    L, psi[running_pos], in_inds[running_pos],
                    out_inds[running_pos]
                )
            end
            if space(L) == space(L_last_time)
                push!(error, abs(norm(L - L_last_time)))
            end
            crit = entanglement_criterion(steps, error)
            steps += 1
        end
        return L
    end

    return L_list
end

# Function to find the list of left projectors L_list
function find_R(
        psi::Vector{<:AbstractTensorMap{E, S}}, in_inds::Vector{Int}, out_inds::Vector{Int},
        entanglement_criterion::stopcrit
    ) where {E, S}
    R_list = map(eachindex(psi)) do i
        R = id(E, domain(psi[i])[in_inds[i]])
        error = Float64[Inf]
        crit = true
        steps = 1
        while crit
            R_last_time = R
            for j in 0:(length(psi) - 1)
                running_pos = mod(i - j - 1, length(psi)) + 1
                R = QR_R(R, psi[running_pos], in_inds[running_pos], out_inds[running_pos])
            end
            if space(R) == space(R_last_time)
                push!(error, abs(norm(R - R_last_time)))
            end
            crit = entanglement_criterion(steps, error)
            steps += 1
        end
        return R
    end
    return R_list
end

# Function to find the projector P_L and P_R
function P_decomp(
        R::TensorMap{E, S, 1, 1}, L::TensorMap{E, S, 1, 1},
        trunc::MatrixAlgebraKit.TruncationStrategy; reversed::Bool = false
    ) where {E, S}
    U, s, V, _ = if reversed
        svd_reversed(L * R; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    else
        svd_trunc(L * R; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    end
    re_sq = pseudopow(s, -0.5)
    PR = R * V' * re_sq
    PL = re_sq * U' * L
    return PR, PL
end

# Function to find the list of projectors
function find_projectors(
        psi::Vector{T}, in_inds::Vector{Int}, out_inds::Vector{Int},
        entanglement_criterion::stopcrit, trunc::TruncationStrategy
    ) where {T <: AbstractTensorMap}
    n = length(psi)
    Ls = find_L(psi, in_inds, out_inds, entanglement_criterion)
    Rs = find_R(psi, out_inds, in_inds, entanglement_criterion)
    PRsPLs = map(1:n) do i
        reversed = isdual(space(psi[i], in_inds[i]))
        return P_decomp(Rs[mod1(i - 1, n)], Ls[i], trunc; reversed)
    end
    PRs = map(Base.Fix2(getindex, 1), PRsPLs)
    PLs = map(Base.Fix2(getindex, 2), PRsPLs)
    return PRs, PLs
end

function MPO_disentangled!(
        psi::Vector{T}, in_inds::Vector{Int}, out_inds::Vector{Int},
        PRs::Vector{TR}, PLs::Vector{TL}
    ) where {
        T <: AbstractTensorMap, TR <: AbstractTensorMap{<:Any, <:Any, 1, 1},
        TL <: AbstractTensorMap{<:Any, <:Any, 1, 1},
    }
    n = length(psi)
    for i in 1:n
        M = length(codomain(psi[i]))
        N = length(domain(psi[i]))
        in_ind = in_inds[i]
        out_ind = out_inds[i]
        permT = (
            (in_ind,),
            (
                reverse(collect(1:(in_ind - 1)))..., collect((M + 1):(M + N))...,
                reverse(collect((in_ind + 1):M))...,
            ),
        )
        permLT = (
            (
                reverse(collect(2:(in_ind + out_ind - 1)))..., 1,
                reverse(collect((in_ind + out_ind + 1):(M + N)))...,
            ), (in_ind + out_ind,),
        )
        permLTR = (
            Tuple(collect(out_ind:(out_ind + M - 1))),
            (
                collect(reverse(1:(out_ind - 1)))..., M + N,
                collect(reverse((out_ind + M):(M + N - 1)))...,
            ),
        )
        LTR = transpose(
            transpose(PLs[i] * transpose(psi[i], permT), permLT) *
                PRs[mod(i, n) + 1], permLTR
        )
        @assert [isdual(space(psi[i], ax)) for ax in 1:numind(psi[i])] ==
            [isdual(space(LTR, ax)) for ax in 1:numind(LTR)]
        psi[i] = LTR
    end
    return
end

function SVD12(
        T::AbstractTensorMap{E, S, 1, 3}, trunc::MatrixAlgebraKit.TruncationStrategy;
        reversed::Bool = false
    ) where {E, S}
    T_trans = transpose(T, ((2, 1), (3, 4)); copy = true)
    U, s, V, e = if reversed
        svd_reversed(T_trans; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    else
        svd_trunc(T_trans; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    end
    @plansor S1[-1; -2 -3] := U[-2 -1; 1] * sqrt(s)[1; -3]
    @plansor S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end

function SVD12(
        T::AbstractTensorMap{E, S, 2, 2}, trunc::MatrixAlgebraKit.TruncationStrategy;
        reversed::Bool = false
    ) where {E, S}
    U, s, V, e = reversed ? svd_reversed(T; trunc = trunc) : svd_trunc(T; trunc = trunc)
    return U * sqrt(s), sqrt(s) * V
end

"""
Given a tensormap `T`, the function computes its singular value decomposition (SVD) and applies soft-thresholding to the singular values σ ↦ max(σ - τ, 0). 
The resulting tensor is reconstructed using the thresholded singular values, and the function also returns the reduced rank (number of non-zero singular values after thresholding) and the nuclear norm (sum of the thresholded singular values).
"""
function singular_value_thresholding(T::TensorMap, tau::Float64)
    U, S, V = svd_trunc(T; trunc = trunctol(; atol = tau))

    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    nuclear_norm = sum(thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U * new_S * V, rank_reduced, nuclear_norm
end
