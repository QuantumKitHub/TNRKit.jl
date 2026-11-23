"""
$(TYPEDEF)

Impurity method for Higher-Order Tensor Renormalization Group, for moments of
**arbitrary order** of (at most two) on-site operators, typically an order parameter (e.g. the magnetisation).

The scheme coarse-grains the pure tensor `T` together with a set of impurity tensors
``S_{a,b}``. Two (possibly unrelated) on-site impurity operators ``A`` and ``B`` are supported.
The initial impurity tensor of order `(a, b)` is the single site tensor with the operator ``A^a B^b`` inserted on that site.

- For a single (e.g. Ising) order parameter ``m``, only ``A = m`` is needed and all orders have the form `(k, 0)`.
- Often ``B`` is the complex conjugate of ``A``, i.e. ``A = m`` and ``B = m^*``, which is needed for complex order parameters
  (e.g. Potts), as for example ``⟨|m|^2⟩ = ⟨m m^*⟩`` is the order `(1, 1)`.
- ``A`` and ``B`` can however be any two operators, e.g. two different fields, in which case `(a, b)` gives the mixed moment ``⟨A^a B^b⟩``.

At each HOTRG (half-)step, the impurity tensors are updated with the binomial rule
```math
S_{a,b} \\leftarrow \\frac{1}{2^{a+b}} \\sum_{i=0}^{a} \\sum_{j=0}^{b} \\binom{a}{i}\\binom{b}{j} \\; R\\left(S_{i,j}, S_{a-i,b-j}\\right)
```
where ``S_{0,0} = T`` is the pure tensor and ``R(X, Y)`` is the HOTRG contraction of the tensors `X` and `Y`
with the isometry of the **pure** tensor. The prefactor ``2^{-(a+b)}`` performs the average over the two halves.
The moments are then obtained as ``⟨A^a B^b⟩ = \\mathrm{Tr}\\, S_{a,b} / \\mathrm{Tr}\\, T``,
which can be extracted from the output of [`run!`](@ref) with [`expectation_value`](@ref),
[`binder_ratio`](@ref) and [`binder_cumulant`](@ref).

!!! info "Which orders need to be provided?"
    To update ``S_{a,b}`` all ``S_{i,j}`` with ``i ≤ a`` and ``j ≤ b`` are needed.
    Therefore the provided set of orders must be *closed downwards*: if `(a, b)` is given,
    `(a - 1, b)` (if `a > 0`) and `(a, b - 1)` (if `b > 0`) must be given as well.
    The cost per step scales as ``\\sum_{(a,b)} (a+1)(b+1)`` HOTRG contractions.

# Constructors
    $(FUNCTIONNAME)(T, T_imp_order1)                                         # first order only
    $(FUNCTIONNAME)(T, [T_imp_order1, T_imp_order2, ..., T_imp_orderN])      # one impurity A, up to order N
    $(FUNCTIONNAME)(T, Dict((a, b) => T_imp_ab, ...))                        # general (two impurities A and B)
    $(FUNCTIONNAME)(T, T_imp_order1_1, T_imp_order1_2, T_imp_order2)         # legacy 2nd order constructor

The legacy constructor is equivalent to `ImpurityHOTRG(T, Dict((1, 0) => T_imp_order1_1, (0, 1) => T_imp_order1_2, (1, 1) => T_imp_order2))`,
i.e. `T_imp_order1_1` holds ``A``, `T_imp_order1_2` holds ``B`` and `T_imp_order2` holds ``AB``.
All impurity tensors must have the same type and spaces as `T`.

# Running the algorithm
    run!(::ImpurityHOTRG, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=ImpurityHOTRG_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2.

The default finalizer returns, at every iteration, a `NamedTuple` `(norm = n, moments = Dict((a, b) => ⟨Āᵃ B̄ᵇ⟩))`.
Use `getindex.(data, 1)` (or `getproperty.(data, :norm)`) for [`free_energy`](@ref) with `scalefactor = 4.0`.

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Examples
```julia
# Ising model: ⟨m⟩, ⟨m²⟩, ⟨m³⟩, ⟨m⁴⟩ and the Binder ratio
β = 0.44
T = classical_ising(Trivial, β)
T_imps = [classical_ising_impurity(β; order = k) for k in 1:4]
scheme = ImpurityHOTRG(T, T_imps)
data = run!(scheme, truncrank(16), maxiter(25))

m2 = expectation_value(data, 2)
U4 = binder_ratio(data)

# Complex order parameter (e.g. complex ϕ⁴) with A = ϕ and B = ϕ†: orders (a, b) up to (2, 2)
T = phi4_complex(Trivial, 6, μ0, λ; T = ComplexF64)
T_imps = Dict((a, b) => phi4_complex_imp(6, μ0, λ, a, b) for a in 0:2, b in 0:2 if (a, b) != (0, 0))
scheme = ImpurityHOTRG(T, T_imps)
```

# Fields

$(TYPEDFIELDS)

# References
* [Morita et al 10.1016/j.cpc.2018.10.014 (2018)](@cite morita2019)
"""
mutable struct ImpurityHOTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "Central Tensor"
    T::TT

    "Impurity tensors, `T_imp[(a, b)]` is the impurity tensor of order (a, b), i.e. with A^a B^b inserted"
    T_imp::Dict{NTuple{2, Int}, TT}

    function ImpurityHOTRG(T::TT, T_imp::Dict{NTuple{2, Int}, TT}) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        @assert !isempty(T_imp) "At least one impurity tensor must be given"
        for ((a, b), T_ab) in T_imp
            @assert a ≥ 0 && b ≥ 0 "Orders must be non-negative, got ($a, $b)"
            @assert (a, b) != (0, 0) "Order (0, 0) is reserved for the pure tensor T"
            @assert space(T_ab) == space(T) "The impurity tensor of order ($a, $b) must have the same spaces as T"
            # The update of S_{a,b} needs all S_{i,j} with i ≤ a and j ≤ b (closed downwards)
            @assert a == 0 || (a - 1, b) == (0, 0) || haskey(T_imp, (a - 1, b)) "Order ($a, $b) requires order ($(a - 1), $b) to be given as well"
            @assert b == 0 || (a, b - 1) == (0, 0) || haskey(T_imp, (a, b - 1)) "Order ($a, $b) requires order ($a, $(b - 1)) to be given as well"
        end
        return new{E, S, TT}(T, T_imp)
    end
end

# General constructor: any dictionary with (a, b) => T_ab
function ImpurityHOTRG(T::TT, T_imp::AbstractDict) where {TT <: AbstractTensorMap}
    return ImpurityHOTRG(T, Dict{NTuple{2, Int}, TT}(T_imp))
end

# One impurity A: T_imp[k] is the impurity tensor of order k (i.e. order (k, 0))
function ImpurityHOTRG(T::TT, T_imp::AbstractVector{TT}) where {TT <: AbstractTensorMap}
    return ImpurityHOTRG(T, Dict((k, 0) => T_k for (k, T_k) in enumerate(T_imp)))
end

# First order only
function ImpurityHOTRG(T::TT, T_imp_order1::TT) where {TT <: AbstractTensorMap}
    return ImpurityHOTRG(T, [T_imp_order1])
end

# Legacy second order constructor (two first order impurities A and B, often ϕ and ϕ†, and their product AB)
function ImpurityHOTRG(T::TT, T_imp_order1_1::TT, T_imp_order1_2::TT, T_imp_order2::TT) where {TT <: AbstractTensorMap}
    return ImpurityHOTRG(T, Dict((1, 0) => T_imp_order1_1, (0, 1) => T_imp_order1_2, (1, 1) => T_imp_order2))
end

# Binomial update of all impurity tensors for one HOTRG (half-)step.
# `step_fun` is `_step_hotrg_x` or `_step_hotrg_y` and `U` the corresponding isometry (of the pure tensor).
# For example for order (2, 0): S_2 <- 1/4 * (R(T, S_2) + 2 R(S_1, S_1) + R(S_2, T)),
# and for order (1, 1):         S_11 <- 1/4 * (R(T, S_11) + R(S_01, S_10) + R(S_10, S_01) + R(S_11, T)).
function _step_impurities(step_fun, T, T_imp, U)
    # Tensor of order (i, j), where (0, 0) is the pure tensor
    tensor_of_order(order) = order == (0, 0) ? T : T_imp[order]

    T_imp_new = empty(T_imp)
    for (a, b) in keys(T_imp)
        T_imp_new[(a, b)] = (1 / 2^(a + b)) * sum(
            binomial(a, i) * binomial(b, j) *
                step_fun(tensor_of_order((i, j)), tensor_of_order((a - i, b - j)), U)
                for i in 0:a for j in 0:b
        )
    end
    return T_imp_new
end

function step!(scheme::ImpurityHOTRG, trunc::TruncationStrategy)
    # Isometries are determined by the pure tensor only
    Uy, _ = _get_hotrg_yproj(scheme.T, scheme.T, trunc)

    T = _step_hotrg_x(scheme.T, scheme.T, Uy)
    T_imp = _step_impurities(_step_hotrg_x, scheme.T, scheme.T_imp, Uy)

    scheme.T = T
    scheme.T_imp = T_imp

    Ux, _ = _get_hotrg_xproj(scheme.T, scheme.T, trunc)

    T = _step_hotrg_y(scheme.T, scheme.T, Ux)
    T_imp = _step_impurities(_step_hotrg_y, scheme.T, scheme.T_imp, Ux)

    scheme.T = T
    scheme.T_imp = T_imp

    return scheme
end

function Base.show(io::IO, scheme::ImpurityHOTRG)
    println(io, "ImpurityHOTRG - Impurity Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    for order in sort!(collect(keys(scheme.T_imp)))
        println(io, "  * T_imp[$order]: $(summary(scheme.T_imp[order]))")
    end
    return nothing
end

run!(scheme::ImpurityHOTRG, trscheme::TruncationStrategy, criterion::stopcrit; kwargs...) = run!(scheme, trscheme, criterion, ImpurityHOTRG_Finalizer; kwargs...)
