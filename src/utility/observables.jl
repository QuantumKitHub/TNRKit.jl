# Observables computed from the output of impurity methods, e.g. `run!(::ImpurityHOTRG, ...)`.
# Every element of `data` is a NamedTuple `(norm = n, moments = Dict((a, b) => ⟨Āᵃ B̄ᵇ⟩))`,
# with Ā and B̄ the lattice averages of the two impurity operators A and B (see `ImpurityHOTRG`).

"""
$(SIGNATURES)

Takes the output `data` of `run!` for an [`ImpurityHOTRG`](@ref) scheme and returns the
expectation value of the `order`-th moment of the (lattice averaged) impurity operators ``A`` and ``B``.

- `order::Integer = k` gives ``⟨A^k⟩`` (one impurity, i.e. order `(k, 0)`).
- `order::NTuple{2, Integer} = (a, b)` gives ``⟨A^a B^b⟩`` (two impurities, often ``B = A^*``, e.g. ϕ and ϕ†).

By default the value after the last iteration is returned. Use the keyword `iteration` to select
an other iteration (`iteration = 1` is the initial state if `finalize_beginning = true`),
or `iteration = :` to obtain the values of all iterations (useful to check convergence).

!!! info
    The value is returned with the scalar type of the tensors. For complex tensors the imaginary
    part of physically real moments (e.g. ``⟨|ϕ|^2⟩ = ⟨ϕϕ^†⟩``) should vanish up to numerical precision.

# Examples
```julia
data = run!(ImpurityHOTRG(T, [T1, T2]), truncrank(16), maxiter(25))
expectation_value(data, 1)                  # ⟨m⟩
expectation_value(data, 2)                  # ⟨m²⟩
expectation_value(data, 2; iteration = :)   # ⟨m²⟩ for every iteration
expectation_value(data, (1, 1))             # ⟨A B⟩, e.g. ⟨|m|²⟩ if B = A* (if this order was computed)
```

See also: [`binder_ratio`](@ref), [`binder_cumulant`](@ref), [`free_energy`](@ref).
"""
function expectation_value(data, order::NTuple{2, Integer}; iteration = lastindex(data))
    return _expectation_value(data[iteration], order)
end
expectation_value(data, order::Integer; kwargs...) = expectation_value(data, (order, 0); kwargs...)

# Single iteration
function _expectation_value(d::NamedTuple, order::NTuple{2, Integer})
    # ⟨A⁰ B⁰⟩ = Tr T / Tr T = 1
    order == (0, 0) && return one(valtype(d.moments))
    haskey(d.moments, order) || throw(
        ArgumentError(
            "The moment of order $order was not computed. Available orders: $(sort!(collect(keys(d.moments))))"
        )
    )
    return d.moments[order]
end
# Multiple iterations
_expectation_value(ds::AbstractVector, order::NTuple{2, Integer}) = map(d -> _expectation_value(d, order), ds)

"""
$(SIGNATURES)

Takes the output `data` of `run!` for an [`ImpurityHOTRG`](@ref) scheme and computes the Binder ratio
```math
U_4 = \\frac{⟨m^4⟩}{⟨m^2⟩^2}
```
as defined in [Morita et al. (2018)](@cite morita2019).

- If the order `(2, 2)` was computed (two impurities), it is assumed that ``B = A^* = m^*`` (complex order parameter) and
  ``U_4 = ⟨|m|^4⟩/⟨|m|^2⟩^2`` is computed from the orders `(2, 2)` and `(1, 1)`.
- Otherwise the orders `(4, 0)` and `(2, 0)` of the first impurity ``A = m`` are used.

Limits: for a real scalar order parameter (e.g. Ising) ``U_4 → 3`` in the disordered phase and ``U_4 → 1``
in the ordered phase. For a complex order parameter (e.g. Potts with ``q > 2``, complex ϕ⁴) ``U_4 → 2``
in the disordered phase.

The keyword `iteration` works as in [`expectation_value`](@ref).

See also: [`binder_cumulant`](@ref), [`expectation_value`](@ref).
"""
function binder_ratio(data; iteration = lastindex(data))
    return _binder_ratio(data[iteration])
end

function _binder_ratio(d::NamedTuple)
    order2, order4 = haskey(d.moments, (2, 2)) ? ((1, 1), (2, 2)) : ((2, 0), (4, 0))
    return _expectation_value(d, order4) / _expectation_value(d, order2)^2
end
_binder_ratio(ds::AbstractVector) = map(_binder_ratio, ds)

"""
$(SIGNATURES)

Takes the output `data` of `run!` for an [`ImpurityHOTRG`](@ref) scheme and computes the Binder cumulant
```math
U_L = 1 - \\frac{U_4}{c}
```
with ``U_4`` the [`binder_ratio`](@ref) and ``c`` the value of ``U_4`` in the disordered phase, such that
``U_L → 0`` in the disordered phase.

By default `c = 3` for one impurity (real order parameter, e.g. Ising: ``U_L = 1 - ⟨m^4⟩/(3⟨m^2⟩^2)``, ``U_L → 2/3`` in the ordered phase)
and `c = 2` when the order `(2, 2)` is available (complex order parameter with ``B = A^*``: ``U_L = 1 - ⟨|m|^4⟩/(2⟨|m|^2⟩^2)``, ``U_L → 1/2`` in the ordered phase).
It can be set manually with the keyword `c`.

The keyword `iteration` works as in [`expectation_value`](@ref).

See also: [`binder_ratio`](@ref), [`expectation_value`](@ref).
"""
function binder_cumulant(data; iteration = lastindex(data), c = haskey(data[end].moments, (2, 2)) ? 2 : 3)
    return 1 .- binder_ratio(data; iteration) ./ c
end
