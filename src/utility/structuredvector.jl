"""
    StructuredVector{E, K, A} <: AbstractVector{E}

A vector whose elements are partitioned into named sectors.  Internally, data
is stored as a flat `AbstractVector{E}` and a `Dict{K, Vector{Int}}` maps each
sector key to the indices that belong to it.

Supports the `AbstractVector` interface (integer indexing, `length`, `eachindex`,
…), sector-based access via `v[sector]`, and the full `Dict` key interface
(`keys`, `in`).  `sort`, `filter`, scalar arithmetic, and element-wise
broadcasting with scalars all preserve the sector structure.

# Constructors

    StructuredVector(sv::SectorVector)
    StructuredVector(dict::Dict{K, <:AbstractVector{E}}) where {K, E}
    StructuredVector(data::AbstractVector{E}, structure::Dict{K, Vector{Int}})

- From a TensorKit `SectorVector`.
- From a dictionary mapping sectors to their data vectors.
- Directly from a flat data array and a sector‑index mapping.
"""
struct StructuredVector{E, K, A <: AbstractVector{E}} <: AbstractVector{E}
    data::A
    structure::Dict{K, Vector{Int}}
end

function StructuredVector(sv::TensorKit.SectorVector)
    structure = Dict(k => collect(r) for (k, r) in sv.structure)
    return StructuredVector(copy(sv.data), structure)
end

function StructuredVector(dict::Dict{K, <:AbstractVector{E}}) where {K, E}
    data = E[]
    structure = Dict{K, Vector{Int}}()
    last_index = 1
    for (key, values) in dict
        isempty(values) && continue
        append!(data, values)
        structure[key] = collect(last_index:(last_index + length(values) - 1))
        last_index += length(values)
    end
    return StructuredVector(data, structure)
end

@inline Base.getindex(v::StructuredVector, i::Int) = getindex(parent(v), i)
@inline Base.getindex(v::StructuredVector{E, K}, keys::K) where {E, K} = parent(v)[v.structure[keys]]
@inline Base.setindex!(v::StructuredVector, val, i::Int) = setindex!(parent(v), val, i)

Base.size(v::StructuredVector, args...) = size(parent(v), args...)
Base.size(v::StructuredVector) = size(parent(v))
Base.copy(v::StructuredVector) = StructuredVector(copy(v.data), v.structure)
Base.parent(v::StructuredVector) = v.data

function Base.sort(v::StructuredVector; kwargs...)
    p = sortperm(v.data; kwargs...)
    inv_p = invperm(p)
    newdict = Dict(k => sort(inv_p[v.structure[k]]) for k in keys(v.structure))
    return StructuredVector(v.data[p], newdict)
end

function Base.filter(f, v::StructuredVector)
    kept_inds = findall(f, parent(v))
    data = parent(v)[kept_inds]
    old_to_new = Dict(old_ind => new_ind for (new_ind, old_ind) in enumerate(kept_inds))

    new_structure = Dict{keytype(v.structure), Vector{Int}}()
    for (sector, inds) in v.structure
        new_structure[sector] = [old_to_new[ind] for ind in inds if haskey(old_to_new, ind)]
    end

    return StructuredVector(data, new_structure)
end

function Base.show(io::IO, v::StructuredVector)
    println(io, "StructuredVector with keys: ", keys(v.structure))
    return print(io, "Data: ", v.data)
end

Base.:*(v::StructuredVector, x::Number) = StructuredVector(v.data .* x, v.structure)
Base.:*(x::Number, v::StructuredVector) = StructuredVector(x .* v.data, v.structure)
Base.:/(v::StructuredVector, x::Number) = StructuredVector(v.data ./ x, v.structure)
Base.:/(x::Number, v::StructuredVector) = StructuredVector(x ./ v.data, v.structure)

Base.keys(v::StructuredVector) = keys(v.structure)

function Base.vcat(v1::StructuredVector{<:Any, K1}, v2::StructuredVector{<:Any, K2}) where {K1, K2}
    new_data = vcat(v1.data, v2.data)
    n1 = length(v1)
    K = promote_type(K1, K2)
    new_structure = Dict{K, Vector{Int}}()
    for (k, inds) in v1.structure
        new_structure[k] = copy(inds)
    end
    for (k, inds) in v2.structure
        adjusted = inds .+ n1
        if haskey(new_structure, k)
            append!(new_structure[k], adjusted)
        else
            new_structure[k] = adjusted
        end
    end
    return StructuredVector(new_data, new_structure)
end

# Julia Base provides a specialized `reduce(::typeof(vcat), A)` that bypasses
# pairwise `vcat` calls and uses `similar` + bulk copy internally.  That path
# does not know about sector structure and would produce a plain `Vector`.
# We override it to fall back to pair-wise reduction which preserves the
# StructuredVector container.
function Base.reduce(::typeof(vcat), A::AbstractVector{<:StructuredVector})
    isempty(A) && throw(ArgumentError("reducing vcat over an empty collection is not supported"))
    return foldl(vcat, A)
end

# -- Broadcasting support -----------------------------------------------------
# Custom broadcast style so that element-wise operations preserve the
# StructuredVector container (and therefore the sector structure).
struct StructuredVectorStyle <: Broadcast.AbstractArrayStyle{1} end
StructuredVectorStyle(::Val{1}) = StructuredVectorStyle()  # parametric resize hook
Base.BroadcastStyle(::Type{<:StructuredVector}) = StructuredVectorStyle()
# Only scalars (and 0-dim arrays) get the StructuredVectorStyle result.
# Mixing with plain arrays is left undefined — it falls back to DefaultArrayStyle.
Base.BroadcastStyle(::StructuredVectorStyle, ::Broadcast.Style{Tuple}) = StructuredVectorStyle()
Base.BroadcastStyle(::Broadcast.Style{Tuple}, ::StructuredVectorStyle) = StructuredVectorStyle()
Base.BroadcastStyle(::StructuredVectorStyle, ::Broadcast.DefaultArrayStyle{0}) = StructuredVectorStyle()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{0}, ::StructuredVectorStyle) = StructuredVectorStyle()

# Walk the broadcast tree to find the StructuredVector that determines the
# output structure.
_find_sv(bc::Broadcast.Broadcasted) = _find_sv(bc.args...)
_find_sv(sv::StructuredVector, rest...) = sv
_find_sv(::Any, rest...) = _find_sv(rest...)
_find_sv() = nothing

function Base.similar(bc::Broadcast.Broadcasted{StructuredVectorStyle}, ::Type{ElType}) where {ElType}
    sv = _find_sv(bc)
    if sv === nothing
        return similar(Array{ElType}, axes(bc))
    end
    return StructuredVector(similar(sv.data, ElType), copy(sv.structure))
end

"""
    mapkeys(f, v::StructuredVector)

Return a new `StructuredVector` with each `structure` key transformed
by a function `f`. The underlying `data` is shared with `v`.
```
"""
function mapkeys(f, v::StructuredVector)
    new_structure = Dict(f(k) => copy(inds) for (k, inds) in v.structure)
    return StructuredVector(v.data, new_structure)
end
