"""
    CFTData{E, I} where {E, I}

A struct to hold conformal data extracted from a TNR scheme.

# Constructors
    CFTData(scheme::TNRScheme; kwargs...)
    CFTData(TA::TensorMap{E, S, 2, 2}; kwargs...)
    CFTData(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; kwargs...)

# Fields
    - `central_charge::Union{E, Missing}`: The central charge of the CFT. Will be `nothing` if not calculated.
    - `modular_parameter::E`: The elementary modular parameter of a square spacetime patch of the CFT.
    - `scaling_dimensions::TensorKit.SectorVector{E, I}`: The scaling dimensions of the CFT, organized in a `TensorKit.SectorVector` where the sectors correspond to different spin sectors (or other quantum numbers) and the data contains the scaling dimensions within those sectors

"""
struct CFTData{E, I}
    "Central charge of the CFT. Will be `missing` if not calculated."
    central_charge::Union{E, Missing}
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
        τ0 = elementary_modular_parameter(T, T)
        Δs = _scaling_dimensions(T)
        return CFTData(missing, τ0, StructuredVector(Δs, Dict([Trivial => collect(eachindex(Δs))])))
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
    τ0 = elementary_modular_parameter(TA′, TB′)
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

# TODO: replace v with solved elementary modular parameter
function _scaling_dimensions(T::TensorMap{E, S, 2, 2}; v = 1, unitcell = 1) where {E, S}
    # stack unitcell copies of T and trace
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

    return unitcell * (1 / (2π * v)) .* log.(data[1] ./ data)
end

"""
The "canonical" normalization constant for loop-TNR tensors,
which is the eigenvalue with largest norm of the 2 x 2 transfer matrix.
"""
function area_term(TA, TB; is_real = true)
    function f0(x)
        @plansor fx[-1 -2] := TA[c -1; 1 m] * x[1 2] * TB[m -2; 2 c]
        @plansor ffx[-1 -2] := TB[c -1; 1 m] * fx[1 2] * TA[m -2; 2 c]
        return ffx
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
    spec(TA::TensorMap, TB::TensorMap, shape::Array; Nh = 25)

Internal function to construct transfer matrices and extract conformal data.

# Parameters

TA, TB: Tensors used to construct the transfer matrix. 
    They can be different from the original tensor in the network.
shape: A triplet `[h, L, x]` describing the geometry of the transfer matrix.
τ0: The modular parameter of a square patch of the original network.
"""
function spec(TA::TensorMap, TB::TensorMap, shape::Array, τ0::Number; Nh = 25)
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
        TA::TensorMap, TB::TensorMap, trunc::TruncationStrategy,
        truncentanglement::TruncationStrategy
    )
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
function MPO_action_2gates(TA::TensorMap, TB::TensorMap, x::TensorMap)
    @tensor fx[-1 -2 -3 -4; 5] := TB[-1 -2; 1 2] * x[1 2 3 4; 5] * TB[-3 -4; 3 4]
    @tensor ffx[-1 -2 -3 -4; 5] := TA[-3 -4; 2 3] * fx[1 2 3 4; 5] *
        TA[-1 -2; 4 1]
    return permute(ffx, ((2, 3, 4, 1), (5,)))
end

"""
When the elementary modular parameter for TA, TB is `τ`,
the transfer matrix has `τ_TM = τ / 4`.
"""
function MPO_action_1x4(TA::TensorMap, TB::TensorMap, x::TensorMap)
    @tensor TTTTx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] * TA[41 -1; 1 12] *
        TB[12 -2; 2 23] *
        TA[23 -3; 3 34] * TB[34 -4; 4 41]
    return TTTTx
end

"""
When the elementary modular parameter for TA, TB is `τ`,
the transfer matrix has `τ_TM = (τ + 1) / 4`.
"""
function MPO_action_1x4_twist(TA::TensorMap, TB::TensorMap, x::TensorMap)
    TTTTx = MPO_action_1x4(TA, TB, x)
    return permute(TTTTx, ((2, 3, 4, 1), (5,)))
end

"""
This renormalization will change the elementary
modular parameter from τ to (τ - 1) / 2.
"""
function reduced_MPO(
        dl::TensorMap, ur::TensorMap, ul::TensorMap, dr::TensorMap,
        trunc::TruncationStrategy
    )
    @plansor temp[-1 -2; -3 -4] :=
        ur[-1; 1 4] * ul[4; 3 -2] * dr[-3; 2 1] * dl[2; -4 3]
    D, U = SVD12(temp, trunc)
    @plansor translate[-1 -2; -3 -4] := U[-2; 1 -4] * D[-1 1; -3]
    return translate
end

# Elementary modular parameter
# ============================
# TODO: one-tensor version
"""
    elementary_modular_parameter(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}) where {E, S}

Extract the elementary modular parameter of one tensor from 2x2 transfer matrices.
"""
function elementary_modular_parameter(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}
    ) where {E, S}
    # vertical (north to south)
    evs_v = _eigsolve_2x2_NtoS(TA, TB)
    # horizontal (east to west)
    evs_h = _eigsolve_2x2_EtoW(TA, TB)
    # diagonal (northeast to southwest)
    evs_a = _eigsolve_2x2_NEtoSW(TA, TB)
    # diagonal (northwest to southeast)
    evs_b = _eigsolve_2x2_NWtoSE(TA, TB)
    # norm²
    v = sqrt(
        log(abs(evs_v[2] / evs_v[1])) /
            log(abs(evs_h[2] / evs_h[1]))
    )
    # phase
    r = log(abs(evs_a[2] / evs_a[1])) /
        log(abs(evs_b[2] / evs_b[1]))
    θ = acos((v^2 + 1) / (2 * v) * (r - 1) / (r + 1))
    return v * cis(θ)
end

function _eigsolve_2x2_NtoS(TA, TB)
    function f0(x)
        @plansor fx[-1 -2] := TA[c -1; 1 m] * x[1 2] * TB[m -2; 2 c]
        @plansor fx[-1 -2] := TB[c -1; 1 m] * fx[1 2] * TA[m -2; 2 c]
        return fx
    end
    x0 = ones(domain(TA, 1) ⊗ domain(TB, 1))
    spec0, _, info = eigsolve(f0, x0, 2, :LM; verbosity = 0)
    if info.converged == 0
        @warn "The area term eigensolver did not converge."
    end
    return spec0
end

function _eigsolve_2x2_EtoW(TA, TB)
    TA′ = permute(TA, ((3, 1), (4, 2)))
    TB′ = permute(TB, ((3, 1), (4, 2)))
    return _eigsolve_2x2_NtoS(TB′, TA′)
end

function _eigsolve_2x2_NEtoSW(TA, TB)
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
    function f0(x)
        @plansor fx[-1 -2 -3 -4] := TB[-2 -3; a b] * x[-1 a b -4]
        @plansor fx[-1 -2 -3 -4] := TA[-1 -2; a b] * fx[a b -3 -4]
        @plansor fx[-1 -2 -3 -4] := TA[-3 -4; a b] * fx[-1 -2 a b]
        @plansor fx[-1 -2 -3 -4] := TB[-4 -1; a b] * fx[-3 a b -2]
        return fx
    end
    x0 = ones(domain(TA, 1) ⊗ domain(TB, 1) ⊗ domain(TB, 2) ⊗ domain(TA, 2))
    spec0, _, info = eigsolve(f0, x0, 2, :LM; verbosity = 0)
    if info.converged == 0
        @warn "The area term eigensolver did not converge."
    end
    return spec0
end

function _eigsolve_2x2_NWtoSE(TA, TB)
    TA′ = permute(TA, ((2, 4), (1, 3)))
    TB′ = permute(TB, ((2, 4), (1, 3)))
    return _eigsolve_2x2_NEtoSW(TB′, TA′)
end
