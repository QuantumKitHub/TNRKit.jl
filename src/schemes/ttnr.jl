"""
    TNOTensor{E,S}

Type alias for a local tensor-network-operator tensor with two physical legs and four
virtual legs, i.e. an `AbstractTensorMap{E,S,2,4}`.

Following the PEPO convention used in PEPSKit, the local tensor is interpreted as
`P_out ⊗ P_in <- N ⊗ E ⊗ S ⊗ W`.
"""
const TNOTensor{E, S} = AbstractTensorMap{E, S, 2, 4}

"""
    $(TYPEDEF)

Matrix-like container for a tensor network operator (TNO) unit cell.

The entries are local operator tensors with two physical indices and four virtual
indices. The container behaves like an `AbstractMatrix`, so it supports `size`,
`axes`, indexing and iteration.

### Constructors
    $(FUNCTIONNAME)(A::AbstractMatrix{<:TNOTensor})
    $(FUNCTIONNAME)(A::TNOTensor; unitcell=(1, 1))

### Fields

$(TYPEDFIELDS)
"""
struct TNO{E, S, TT <: TNOTensor{E, S}} <: AbstractMatrix{TT}
    "Matrix of local TNO tensors over the unit cell."
    A::Matrix{TT}

    function TNO(A::AbstractMatrix{TT}) where {E, S, TT <: TNOTensor{E, S}}
        size(A, 1) > 0 || throw(ArgumentError("The unit cell must have a positive number of rows."))
        size(A, 2) > 0 || throw(ArgumentError("The unit cell must have a positive number of columns."))
        return new{E, S, TT}(Matrix(A))
    end
end

function TNO(A::TT; unitcell::Tuple{Int, Int} = (1, 1)) where {E, S, TT <: TNOTensor{E, S}}
    rows, cols = unitcell
    rows > 0 || throw(ArgumentError("The unit cell must have a positive number of rows."))
    cols > 0 || throw(ArgumentError("The unit cell must have a positive number of columns."))
    return TNO([copy(A) for _ in 1:rows, _ in 1:cols])
end

Base.IndexStyle(::Type{<:TNO}) = IndexCartesian()
Base.eltype(::Type{TNO{E, S, TT}}) where {E, S, TT} = TT
Base.size(tno::TNO) = size(tno.A)
Base.axes(tno::TNO) = axes(tno.A)
Base.getindex(tno::TNO, i::Int, j::Int) = tno.A[i, j]
Base.setindex!(tno::TNO, value, i::Int, j::Int) = setindex!(tno.A, value, i, j)
Base.copy(tno::TNO) = TNO(copy.(tno.A))

"""
    $(TYPEDEF)

Minimal storage object for thermal tensor network renormalization on a square-lattice
tensor-network-operator unit cell.

### Constructors
    $(FUNCTIONNAME)(A::AbstractMatrix{<:AbstractTensorMap})

### Running the algorithm
    run!(::ThermalTNR, A::AbstractMatrix{<:AbstractTensorMap}, trunc::TruncationStrategy, criterion::stopcrit[
              , finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

### Fields

$(TYPEDFIELDS)

### References
* [Ueda et al. (2025)](@cite ueda_2025)
"""
mutable struct ThermalTNR{E, S} <: TNRScheme{E, S}
    "Internal tensor network operator stored in the current TTNR layer."
    T::TNO{E, S}

    function ThermalTNR(T::TT) where {E, S, TT <: TNO{E, S}}
        return new{E, S}(T)
    end
end

function ThermalTNR(A::AbstractMatrix{TT}) where {E, S, TT <: TNOTensor{E, S}}
    return ThermalTNR(TNO(A))
end

function ThermalTNR(A::TT; unitcell::Tuple{Int, Int} = (1, 1)) where {E, S, TT <: TNOTensor{E, S}}
    return ThermalTNR(TNO(A; unitcell = unitcell))
end

Base.copy(scheme::ThermalTNR) = ThermalTNR(copy(scheme.T))

const _TNO_NORTH_AXIS = 3
const _TNO_EAST_AXIS = 4
const _TNO_SOUTH_AXIS = 5
const _TNO_WEST_AXIS = 6

@inline _right_index(j, ncols) = mod1(j + 1, ncols)
@inline _down_index(i, nrows) = mod1(i + 1, nrows)
@inline _up_index(i, nrows) = mod1(i - 1, nrows)

function _check_tno_bond_structure(tno::TNO)
    nrows, ncols = size(tno)

    for i in 1:nrows, j in 1:ncols
        T = tno[i, j]
        T_right = tno[i, _right_index(j, ncols)]
        T_down = tno[_down_index(i, nrows), j]

        space(T, _TNO_EAST_AXIS) == space(T_right, _TNO_WEST_AXIS)' ||
            throw(ArgumentError("East-west bond mismatch between sites ($i, $j) and ($i, $(_right_index(j, ncols)))."))

        space(T, _TNO_SOUTH_AXIS) == space(T_down, _TNO_NORTH_AXIS)' ||
            throw(ArgumentError("North-south bond mismatch between sites ($i, $j) and ($(_down_index(i, nrows)), $j)."))
    end

    return nothing
end

function _check_tno_compatibility(top::TNO, bottom::TNO)
    size(top) == size(bottom) ||
        throw(ArgumentError("The two TNOs must have the same unit-cell dimensions."))

    _check_tno_bond_structure(top)
    _check_tno_bond_structure(bottom)

    for i in axes(top, 1), j in axes(top, 2)
        T_top = top[i, j]
        T_bottom = bottom[i, j]

        space(T_top, 2) == space(T_bottom, 1)' ||
            throw(ArgumentError("Physical output/input mismatch at site ($i, $j)."))
    end

    return nothing
end

function QR_two_pepo_left(O1::TNOTensor, O2::TNOTensor, ind::Int)
    pb = (1, ind)
    p, q1 = ind_pair(O1, pb)
    _, Rb = left_orth(permute(O1, (q1, p)))

    pt = (2, ind)
    p, q2 = ind_pair(O2, pt)
    _, Rt = left_orth(permute(O2, (q2, p)))

    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = left_orth(permute(M, ((3, 4), (1, 2))))
    return R
end

function QR_two_pepo_right(O1::TNOTensor, O2::TNOTensor, ind::Int)
    pb = (1, ind)
    p, q1 = ind_pair(O1, pb)
    Rb, _ = right_orth(permute(O1, (p, q1)))

    pt = (2, ind)
    p, q2 = ind_pair(O2, pt)
    Rt, _ = right_orth(permute(O2, (p, q2)))

    @tensor M[-1 -2; -3 -4] := Rt[1 -1; -3] * Rb[1 -2; -4]
    R, _ = right_orth(permute(M, ((1, 2), (3, 4))))
    return R
end

function QR_two_pepo(O1::TNOTensor, O2::TNOTensor, ind::Int; side = :left)
    if side == :left
        return QR_two_pepo_left(O1, O2, ind)
    elseif side == :right
        return QR_two_pepo_right(O1, O2, ind)
    else
        throw(ArgumentError("side should be :left or :right"))
    end
end

function R1R2(
        A1::TNOTensor, A2::TNOTensor, A3::TNOTensor, A4::TNOTensor,
        ind1::Int, ind2::Int; check_space = true
    )
    RA1 = QR_two_pepo_left(A1, A2, ind1)
    RA2 = QR_two_pepo_right(A3, A4, ind2)
    if check_space && domain(RA1) != codomain(RA2)
        throw(ArgumentError("space mismatch"))
    end
    return RA1, RA2
end

function find_P1P2(
        A1::TNOTensor, A2::TNOTensor, A3::TNOTensor, A4::TNOTensor,
        p1::Int, p2::Int, trunc::TruncationStrategy; check_space = true
    )
    R1, R2 = R1R2(A1, A2, A3, A4, p1, p2; check_space = check_space)
    return oblique_projector(R1, R2, trunc)
end

function _bond_projectors(top::TNO, bottom::TNO, trunc::TruncationStrategy)
    nrows, ncols = size(top)
    north_proj = Matrix{Any}(undef, nrows, ncols)
    east_proj = Matrix{Any}(undef, nrows, ncols)
    south_proj = Matrix{Any}(undef, nrows, ncols)
    west_proj = Matrix{Any}(undef, nrows, ncols)

    for i in 1:nrows, j in 1:ncols
        inorth = _up_index(i, nrows)
        jeast = _right_index(j, ncols)

        Pnorth, Psouth = find_P1P2(
            top[i, j], bottom[i, j], top[inorth, j], bottom[inorth, j], 3, 5, trunc
        )
        Peast, Pwest = find_P1P2(
            top[i, j], bottom[i, j], top[i, jeast], bottom[i, jeast], 4, 6, trunc
        )

        north_proj[i, j] = Pnorth
        south_proj[inorth, j] = Psouth
        east_proj[i, j] = Peast
        west_proj[i, jeast] = Pwest
    end

    return north_proj, east_proj, south_proj, west_proj
end

function _compose_local_tno(top::TNOTensor, bottom::TNOTensor, Pnorth, Peast, Psouth, Pwest)

    @tensor merged[-1 -2; -3 -4 -5 -6] :=
        top[1 -2; 7 8 9 10] *
        bottom[-1 1; 3 4 5 6] *
        Pwest[-6; 6 10] *
        Pnorth[3 7; -3] *
        Peast[4 8; -4] *
        Psouth[-5; 5 9]

    return merged
end

"""
    apply!(top::TNO, bottom::TNO, trunc::TruncationStrategy)

Compose two TNO layers sitewise. The physical input leg of `top` is contracted with the
physical output leg of `bottom`, after which each pair of corresponding virtual bonds is
merged into a single fused bond.

The result is returned as a new `TNO` with the same unit-cell shape.
"""
function apply!(top::TNO, bottom::TNO, trunc::TruncationStrategy)
    _check_tno_compatibility(top, bottom)
    north_proj, east_proj, south_proj, west_proj = _bond_projectors(top, bottom, trunc)
    merged = [
        _compose_local_tno(
                top[i, j], bottom[i, j],
                north_proj[i, j], east_proj[i, j], south_proj[i, j], west_proj[i, j]
            ) for
            i in axes(top, 1), j in axes(top, 2)
    ]
    return TNO(merged)
end

function apply!(top::ThermalTNR, bottom::ThermalTNR, trunc::TruncationStrategy)
    top.T = apply!(top.T, bottom.T, trunc)
    return top
end

function step!(scheme::ThermalTNR, layer::TNO, trunc::TruncationStrategy)
    scheme.T = apply!(scheme.T, layer, trunc)
    return scheme
end

function step!(
        scheme::ThermalTNR, layer::AbstractMatrix{TT}, trunc::TruncationStrategy
    ) where {E, S, TT <: TNOTensor{E, S}}
    return step!(scheme, TNO(layer), trunc)
end

function step!(scheme::ThermalTNR, layer::ThermalTNR, trunc::TruncationStrategy)
    return step!(scheme, layer.T, trunc)
end

function run!(
        scheme::ThermalTNR, layer::TNO, trscheme::TruncationStrategy,
        criterion::stopcrit, finalizer::Finalizer{E};
        finalize_beginning = true, verbosity = 1
    ) where {E}
    data = Vector{E}()

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, finalizer.f!(scheme))
        end

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, layer, trscheme)
            push!(data, finalizer.f!(scheme))

            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

function run!(
        scheme::ThermalTNR, layer::ThermalTNR, trscheme::TruncationStrategy,
        criterion::stopcrit, finalizer::Finalizer{E}; kwargs...
    ) where {E}
    return run!(scheme, layer.T, trscheme, criterion, finalizer; kwargs...)
end

function run!(
        scheme::ThermalTNR, layer::AbstractMatrix{TT}, trscheme::TruncationStrategy,
        criterion::stopcrit, finalizer::Finalizer{E}; kwargs...
    ) where {E, S, TT <: TNOTensor{E, S}}
    return run!(scheme, TNO(layer), trscheme, criterion, finalizer; kwargs...)
end

function run!(scheme::ThermalTNR, layer, trscheme, criterion; kwargs...)
    return run!(scheme, layer, trscheme, criterion, default_Finalizer; kwargs...)
end

function run!(scheme::ThermalTNR, trscheme::TruncationStrategy, criterion::stopcrit; kwargs...)
    scheme_original = copy(scheme)
    return run!(scheme, scheme_original, trscheme, criterion; kwargs...)
end

function Base.show(io::IO, scheme::ThermalTNR)
    println(io, "ThermalTNR - Thermal Tensor Network Renormalization")
    println(io, "  * unit cell: $(size(scheme.T))")
    println(io, "  * tensor: $(summary(scheme.T[1, 1]))")
    return nothing
end
