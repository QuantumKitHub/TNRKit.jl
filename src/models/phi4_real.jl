#####################################
#       HELPER FUNCTIONS            #
#####################################

# Boltzmann weight of a single bond between two sites with fields ϕ1 and ϕ2.
# The on-site terms are distributed evenly over the four bonds attached to each site.
function f_real(ϕ1::Float64, ϕ2::Float64, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    return exp(
        -1 / 2 * (ϕ1 - ϕ2)^2
            - μ0 / 8 * (ϕ1^2 + ϕ2^2)
            - λ / 16 * (ϕ1^4 + ϕ2^4)
            + h / 4 * (ϕ1 + ϕ2)
    )
end

# Bond weight `f_real` evaluated on all pairs of Gauss-Hermite nodes `ys`, as a K × K matrix.
function fmatrix_real(ys::Vector{Float64}, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    matrix = [f_real(ϕ1, ϕ2, μ0, λ, h) for ϕ1 in ys, ϕ2 in ys]
    return TensorMap(matrix, ℂ^length(ys) ← ℂ^length(ys))
end

# Moments M[n + 1] = ∫ ϕⁿ exp(-(2 + μ0 / 2) ϕ² - λ / 4 ϕ⁴) dϕ over the real line for n = 0, …, nmax.
# Only even n give non-zero values.
function moments_real(nmax::Integer, μ0::Float64, λ::Float64)
    λ >= 0 || throw(ArgumentError("The integrals diverge for λ < 0"))
    a = 2 + μ0 / 2
    b = λ / 4
    return map(0:nmax) do n
        isodd(n) && return 0.0
        # Evaluate in log-space to avoid Inf * 0 = NaN for large ϕ; the integrand is even in ϕ
        integrand(ϕ) = exp(n * log(ϕ) - a * ϕ^2 - b * ϕ^4)
        return 2 * first(quadgk(integrand, 0.0, Inf; rtol = 1.0e-8, maxevals = 10^7))
    end
end

#####################################
#   Gauss-Hermite quadrature        #
#####################################

# On-site insertions g(ϕ) for the partition function and the impurity tensors.
const PHI4_REAL_INSERTIONS = (
    one = _ -> 1,
    ϕ = x -> x,
    ϕ2 = x -> x^2,
)

# Builds one Gauss-Hermite tensor per on-site insertion in `insertions`.
# All tensors share the SVD of the bond matrix, so computing them together is cheaper.
function phi4_real_quadrature_tensors(K::Integer, μ0::Float64, λ::Float64, h::Float64, insertions::Tuple; T::Type{<:Number})
    ys, ws = gausshermite(K)

    # Quadrature weights; exp(y²) undoes the Gauss-Hermite weight function
    weights = map(g -> [g(ys[p]) * ws[p] * exp(ys[p]^2) for p in 1:K], insertions)

    U, S, V = map(x -> convert(Array, x), svd_compact!(fmatrix_real(ys, μ0, λ, h)))
    sqrtS = sqrt.(diag(S))

    arrays = map(_ -> zeros(T, K, K, K, K), insertions)

    @threads for i in 1:K
        for j in i:K, k in 1:K, l in k:K
            factor = sqrtS[i] * sqrtS[j] * sqrtS[k] * sqrtS[l]
            sums = map(w -> zero(promote_type(eltype(w), Float64)), weights)
            for p in 1:K
                x = factor * U[p, i] * U[p, j] * V[k, p] * V[l, p]
                sums = map((s, w) -> s + w[p] * x, sums, weights)
            end

            # Fill the symmetric counterparts under i <-> j and k <-> l
            for I in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k))
                foreach((A, s) -> A[I...] = s, arrays, sums)
            end
        end
    end

    return map(A -> TensorMap(A, ℂ^K ⊗ ℂ^K ← ℂ^K ⊗ ℂ^K), arrays)
end

#####################################
#       TENSOR FUNCTIONS            #
#####################################

"""
    phi4_real(K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; kwargs...)
    phi4_real(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; T::Type{<:Number} = Float64)
    phi4_real(::Type{Z2Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)

Constructs the partition function tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation (and bond dimension) `K`, bare mass ``µ_0^2`` `μ0`, interaction constant `λ` and external field `h`.

Compatible with no symmetry or with explicit ℤ₂ symmetry on each of its spaces.
Defaults to ℤ₂ symmetry and `h = 0` if the symmetry type and magnetic field are not provided.

# Arguments
- `K::Integer`: Approximation parameter.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.
- `h::Float64`: External field (default is 0).

# Approximation parameter `K`
## Trivial (no symmetry)
The tensor is constructed by performing a [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature) to approximate the integrals in the partition function.
The bond dimension is equal to the number of quadrature points `K`.

## ℤ₂ symmetry
The tensor is constructed by Taylor expanding the mixed sites term in the partition function.
The order of the Taylor expansion is `K`, which has to be even, and the bond dimension is `K/2` for the even and odd sectors each (`K` in total).
Not compatible with a non-zero magnetic field, as the magnetic field breaks the ℤ₂ symmetry.

# Examples
```julia
    phi4_real(10, -1.0, 1.0) # default ℤ₂ symmetry, h = 0
    phi4_real(Trivial, 10, -1.0, 1.0, 1.0) # no symmetry with magnetic field
```

!!! info
    When studying this model with impurities, the tensor without symmetry should be constructed, as the impurity breaks the ℤ₂ symmetry.

# References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)
* [Delcamp et. al. Phys. Rev. Research 2, 033278 (2020)](@cite delcamp2020)

See also: [`phi4_real_imp1`](@ref), [`phi4_real_imp2`](@ref).
"""
function phi4_real(K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; kwargs...)
    return phi4_real(Z2Irrep, K, μ0, λ, h; kwargs...)
end
function phi4_real(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; T::Type{<:Number} = Float64)
    return only(phi4_real_quadrature_tensors(K, μ0, λ, h, (PHI4_REAL_INSERTIONS.one,); T))
end
function phi4_real(::Type{Z2Irrep}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; T::Type{<:Number} = Float64)
    iszero(h) || throw(ArgumentError("External magnetic field is not compatible with ℤ₂ symmetry"))
    iseven(K) || throw(ArgumentError("K must be even to split into even/odd groups"))

    # ϕ appears with a total power of at most 4(K - 1)
    moments = moments_real(4 * (K - 1), μ0, λ)
    logfact = logfactorial.(0:(K - 1))

    t = zeros(T, K, K, K, K)
    perms = collect(permutations(1:4))

    # The tensor is fully symmetric, so loop only over sorted tuples
    for s1 in 0:(K - 1), s2 in s1:(K - 1), s3 in s2:(K - 1), s4 in s3:(K - 1)
        # Answer is zero if the total power of ϕ is odd
        n = s1 + s2 + s3 + s4
        isodd(n) && continue

        logdenom = 0.5 * (logfact[s1 + 1] + logfact[s2 + 1] + logfact[s3 + 1] + logfact[s4 + 1])
        val = moments[n + 1] / exp(logdenom)

        # Fill all 24 symmetric permutations
        idxs = (s1 + 1, s2 + 1, s3 + 1, s4 + 1)
        for p in perms
            t[ntuple(n -> idxs[p[n]], 4)...] = val
        end
    end

    # Make it block diagonal: even powers first, then odd powers
    perm = vcat(1:2:K, 2:2:K)
    t = t[perm, perm, perm, perm]

    V = Z2Space(0 => K ÷ 2, 1 => K ÷ 2)
    return TensorMap(t, V ⊗ V ← V ⊗ V)
end

"""
    phi4_real_imp1([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; T::Type{<:Number} = Float64)

Constructs the impurity tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation (and bond dimension) `K`, bare mass ``µ_0^2`` `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.
- `h::Float64`: External field (default is 0).

# Examples
```julia
    phi4_real_imp1(10, -1.0, 1.0, 0.0)
```

# References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_real`](@ref), [`phi4_real_imp2`](@ref).
"""
function phi4_real_imp1(K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; kwargs...)
    return phi4_real_imp1(Trivial, K, μ0, λ, h; kwargs...)
end
function phi4_real_imp1(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; T::Type{<:Number} = Float64)
    return only(phi4_real_quadrature_tensors(K, μ0, λ, h, (PHI4_REAL_INSERTIONS.ϕ,); T))
end

"""
    phi4_real_imp2([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; T::Type{<:Number} = Float64)

Constructs the impurity tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation (and bond dimension) `K`, bare mass ``µ_0^2`` `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ^2 operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.
- `h::Float64`: External field (default is 0).

# Examples
```julia
    phi4_real_imp2(10, -1.0, 1.0, 0.0)
```

# References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_real`](@ref), [`phi4_real_imp1`](@ref).
"""
function phi4_real_imp2(K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; kwargs...)
    return phi4_real_imp2(Trivial, K, μ0, λ, h; kwargs...)
end
function phi4_real_imp2(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0; T::Type{<:Number} = Float64)
    return only(phi4_real_quadrature_tensors(K, μ0, λ, h, (PHI4_REAL_INSERTIONS.ϕ2,); T))
end
