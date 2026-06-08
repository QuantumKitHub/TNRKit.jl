struct StructuredVector{E, K, V, A <: AbstractVector{E}} <: AbstractVector{E}
    data::A
    structure::Dict{K, V}
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
Base.:/(v::StructuredVector, x) = StructuredVector(v.data ./ x, v.structure)
Base.:/(x, v::StructuredVector) = StructuredVector(x ./ v.data, v.structure)
