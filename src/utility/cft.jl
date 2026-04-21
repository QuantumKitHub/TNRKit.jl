# Extensions on top of TensorKit.SectorVector
Base.broadcasted(f, v::TensorKit.SectorVector) = TensorKit.SectorVector(broadcast(f, parent(v)), v.structure)
Base.broadcasted(f, v::TensorKit.SectorVector, a) = TensorKit.SectorVector(broadcast(f, parent(v), a), v.structure)
Base.broadcasted(f, a, v::TensorKit.SectorVector) = TensorKit.SectorVector(broadcast(f, a, parent(v)), v.structure)
function Base.broadcasted(f, v1::TensorKit.SectorVector, v2::TensorKit.SectorVector)
    if v1.structure != v2.structure
        throw(ArgumentError("Cannot broadcast two SectorVectors with different structures"))
    end
    return TensorKit.SectorVector(broadcast(f, parent(v1), parent(v2)), v1.structure)
end

function Base.filter(f, v::TensorKit.SectorVector)
    data = copy(parent(v))
    structure = copy(v.structure)

    kept_inds = findall(f, parent(v))
    sectors = keys(structure)
    for (i, sector) in enumerate(sectors)
        structure[sector] = i == 1 ? (1:findlast(x -> x <= structure[sector].stop, kept_inds)) : ((structure[sectors[i - 1]].stop + 1):findlast(x -> x <= structure[sector].stop, kept_inds))
    end
    data = data[kept_inds]
    return TensorKit.SectorVector(data, structure)
end

function Base.sort(v::TensorKit.SectorVector; kwargs...)
    # sort within the sectors, then concatenate the data and update the structure
    # Ideally this would sort the total data, but sectorvectors are only compatible with
    # structures that contain unitranges, we cannot interweave the data of different sectors.
    data = copy(parent(v))
    newdata = similar(data)
    structure = copy(v.structure)
    sectors = keys(structure)
    for sector in sectors
        newdata[structure[sector]] = sort(data[structure[sector]]; kwargs...)
    end
    return TensorKit.SectorVector(newdata, structure)
end

Base.:*(a::Number, v::TensorKit.SectorVector) = scale(v, a)
Base.:*(v::TensorKit.SectorVector, a::Number) = scale(v, a)

"""
    CFTData{E, I} where {E, I}

A struct to hold conformal data extracted from a TNR scheme.

### Constructors


### Fields
    - `central_charge::Union{E, Missing}`: The central charge of the CFT. Will be `nothing` if not calculated.
    - `scaling_dimensions::TensorKit.SectorVector{E, I}`: The scaling dimensions of the CFT, organized in a `TensorKit.SectorVector` where the sectors correspond to different spin sectors (or other quantum numbers) and the data contains the scaling dimensions within those sectors

"""
struct CFTData{E, I}
    "Central charge of the CFT. Will be `nothing` if not calculated."
    central_charge::Union{E, Missing}

    "Scaling dimensions of the CFT."
    scaling_dimensions::TensorKit.SectorVector{E, I}
end

function Base.show(io::IO, data::CFTData)
    println(io, "CFTData")
    println(io, "  * central charge: $(data.central_charge)")
    println(io, "  * scaling dimensions: $(data.scaling_dimensions)")
    return nothing
end

function CFTData(T::TensorMap{E, S, 2, 2}; shape = [sqrt(2), 2 * sqrt(2), 0], kwargs...) where {E, S}
    if shape == [1, 1, 0] # trivial implementation
        return CFTData(missing, _scaling_dimensions(T))
    else
        CFTData(T, T; shape, kwargs...)
    end
end
CFTData(scheme::TNRScheme; kwargs...) = CFTData(scheme.T; kwargs...) # simple 1-site unitcell schemes
CFTData(scheme::LoopTNR; kwargs...) = CFTData(scheme.TA, scheme.TB; kwargs...) # simple 1-site unitcell schemes
function CFTData(scheme::BTRG; kwargs...) # merge bond tensors into central tensor
    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    return CFTData(T_unit; kwargs...)
end

# Main implementation, two-site unitcell
function CFTData(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}; shape = [sqrt(2), 2 * sqrt(2), 0], trunc = truncrank(16), truncentanglement = trunctol(; rtol = 1.0e-14)) where {E, S}
    if shape == [1, 1, 0]
        throw(ArgumentError("The shape [1, 1, 0] is not compatible with a two-site unit cell."))
    elseif (shape ≈ [sqrt(2), 2 * sqrt(2), 0]) || (shape == [1, 4, 1]) # these shapes need no truncation
        norm_const = area_term(TA, TB)^(1 / 4) # canonical normalisation constant
        return spec(TA / norm_const, TB / norm_const, shape)
    elseif (shape == [1, 8, 1]) || (shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)])

        norm_const = area_term(TA, TB)^(1 / 4) # canonical normalisation constant

        dl, ur, ul, dr = MPO_opt(
            TA / norm_const, TB / norm_const, trunc, truncentanglement
        )
        T = reduced_MPO(dl, ur, ul, dr, trunc)
        return spec(T, T, shape)
    else
        throw(ArgumentError("Shape $shape is not implemented."))
    end
end

# Trivial diagonalisation of the transfer matrix. Currently the v and unitcell are not acessible from the outside.
# The user should really be using the other shapes anyways.
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
which is the eigenvalue with largest real part of the 2 x 2 transfer matrix.
"""
function area_term(A, B; is_real = true)
    a_in = domain(A)[1]
    b_in = domain(B)[1]
    x0 = ones(a_in ⊗ b_in)

    function f0(x)
        @plansor fx[-1 -2] := A[c -1; 1 m] * x[1 2] * B[m -2; 2 c]
        @plansor ffx[-1 -2] := B[c -1; 1 m] * fx[1 2] * A[m -2; 2 c]
        return ffx
    end

    spec0, _, info = eigsolve(f0, x0, 1, :LR; verbosity = 0)
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
function spec(TA::TensorMap, TB::TensorMap, shape::Array; Nh = 25)
    area = shape[1] * shape[2]
    Imτ = shape[1] / shape[2]
    relative_shift = shape[3] / shape[1]

    I = sectortype(TA)
    𝔽 = field(TA)
    if BraidingStyle(I) != Bosonic()
        throw(ArgumentError("Sectors with non-Bosonic charge $I has not been implemented"))
    end

    xspace, f = if shape ≈ [1, 4, 1]
        domain(TA)[1] ⊗ domain(TB)[1] ⊗ domain(TA)[1] ⊗ domain(TB)[1],
            MPO_action_1x4_twist
    elseif shape ≈ [1, 8, 1]
        domain(TA)[1] ⊗ domain(TB)[1] ⊗ domain(TA)[1] ⊗ domain(TB)[1],
            MPO_action_1x4
    elseif shape ≈ [sqrt(2), 2 * sqrt(2), 0] ||
            shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
        domain(TB) ⊗ domain(TB), MPO_action_2gates
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

    norm_const_0 = spec_sector[one(I)][1]
    central_charge = 6 / pi / (Imτ - area / 4) * log(norm_const_0)

    # Construct a SectorVector from the data of the different sectors
    data = ComplexF64[]
    structure = TensorKit.SectorDict{sectortype(xspace), UnitRange{Int}}()
    last_index = 1
    for charge in sectors(fuse(xspace))
        DeltaS = -1 / (2 * pi * Imτ) * log.(spec_sector[charge] / norm_const_0)
        if !(relative_shift ≈ 0)
            push!(data, (real.(DeltaS) + imag.(DeltaS) / relative_shift * im)...)
            structure[charge] = last_index:(last_index + length(DeltaS) - 1)
        else
            push!(data, real.(DeltaS)...)
            structure[charge] = last_index:(last_index + length(DeltaS) - 1)
        end
        last_index += length(DeltaS)
    end

    sv = TensorKit.SectorVector(data, structure)
    sv = sort(sv; by = real)
    sv = filter(x -> real(x) ≤ 1.0e16, sv)

    return CFTData(central_charge, sv)
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
function MPO_action_2gates(TA::TensorMap, TB::TensorMap, x::TensorMap)
    @tensor fx[-1 -2 -3 -4; 5] := TB[-1 -2; 1 2] * x[1 2 3 4; 5] * TB[-3 -4; 3 4]
    @tensor ffx[-1 -2 -3 -4; 5] := TA[-3 -4; 2 3] * fx[1 2 3 4; 5] *
        TA[-1 -2; 4 1]
    return permute(ffx, ((2, 3, 4, 1), (5,)))
end

function MPO_action_1x4(TA::TensorMap, TB::TensorMap, x::TensorMap)
    @tensor TTTTx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] * TA[41 -1; 1 12] *
        TB[12 -2; 2 23] *
        TA[23 -3; 3 34] * TB[34 -4; 4 41]
    return TTTTx
end

function MPO_action_1x4_twist(TA::TensorMap, TB::TensorMap, x::TensorMap)
    TTTTx = MPO_action_1x4(TA, TB, x)
    return permute(TTTTx, ((2, 3, 4, 1), (5,)))
end

function reduced_MPO(
        dl::TensorMap, ur::TensorMap, ul::TensorMap, dr::TensorMap,
        trunc::TruncationStrategy
    )
    @plansor temp[-1 -2; -3 -4] := ur[-1; 1 4] *
        ul[4; 3 -2] *
        dr[-3; 2 1] * dl[2; -4 3]
    D, U = SVD12(temp, trunc)
    @plansor translate[-1 -2; -3 -4] := U[-2; 1 -4] * D[-1 1; -3]
    return translate
end
