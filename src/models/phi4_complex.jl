#####################################
#       HELPER FUNCTIONS            #
#####################################

# Boltzmann weight of a single bond between two sites with fields ϕ₁ = ℝϕ1 + iℂϕ1 and ϕ₂ = ℝϕ2 + iℂϕ2.
# The on-site terms are distributed evenly over the four bonds attached to each site.
function f_complex(ℝϕ1::Float64, ℂϕ1::Float64, ℝϕ2::Float64, ℂϕ2::Float64, μ0::Float64, λ::Float64)
    return exp(
        -1 / 2 * ((ℝϕ1 - ℝϕ2)^2 + (ℂϕ1 - ℂϕ2)^2)
            - μ0 / 8 * (ℝϕ1^2 + ℂϕ1^2 + ℝϕ2^2 + ℂϕ2^2)
            - λ / 16 * ((ℝϕ1^2 + ℂϕ1^2)^2 + (ℝϕ2^2 + ℂϕ2^2)^2)
    )
end

# Bond weight `f_complex` evaluated on all pairs of Gauss-Hermite nodes `ys`, as a K² × K² matrix.
function fmatrix_complex(ys::Vector{Float64}, μ0::Float64, λ::Float64)
    K = length(ys)
    matrix = zeros(K^2, K^2)
    @threads for i in 1:K
        for j in i:K, k in 1:K, l in 1:K
            idx1 = (i - 1) * K + j
            idx2 = (k - 1) * K + l
            if idx2 >= idx1  # only compute upper triangle
                val = f_complex(ys[i], ys[j], ys[k], ys[l], μ0, λ)
                matrix[idx1, idx2] = val
                matrix[idx2, idx1] = val  # symmetric counterpart

                # Based on the simultaneous symmetry of (i,j)<->(j,i) and (k,l)<->(l,k)
                idx3 = (j - 1) * K + i
                idx4 = (l - 1) * K + k
                matrix[idx3, idx4] = val
                matrix[idx4, idx3] = val  # symmetric counterpart
            end
        end
    end
    return TensorMap(matrix, ℂ^(K^2) ← ℂ^(K^2))
end

# Radial moments M[n + 1] = ∫₀^∞ rⁿ exp(-(2 + μ0 / 2) r² - λ / 4 r⁴) dr for n = 0, …, nmax.
function radial_moments_complex(nmax::Integer, μ0::Float64, λ::Float64)
    λ >= 0 || throw(ArgumentError("The radial integrals diverge for λ < 0"))
    a = 2 + μ0 / 2
    b = λ / 4
    return map(0:nmax) do n
        # Evaluate in log-space to avoid Inf * 0 = NaN for large r
        integrand(r) = exp(n * log(r) - a * r^2 - b * r^4)
        return first(quadgk(integrand, 0.0, Inf; rtol = 1.0e-8, maxevals = 10^7))
    end
end

#####################################
#   Gauss-Hermite quadrature        #
#####################################

# On-site insertions g(Re ϕ, Im ϕ) for the partition function and the impurity tensors.
const PHI4_COMPLEX_INSERTIONS = (
    one = (_, _) -> 1,
    ϕ = (x, y) -> x + y * im,
    ϕdag = (x, y) -> x - y * im,
    ϕabs = (x, y) -> sqrt(x^2 + y^2),
    ϕ2 = (x, y) -> x^2 + y^2,
)

# Builds one Gauss-Hermite tensor per on-site insertion in `insertions`.
# All tensors share the SVD of the bond matrix, so computing them together is cheaper.
function phi4_complex_quadrature_tensors(K::Integer, μ0::Float64, λ::Float64, insertions::Tuple; T::Type{<:Number})
    ys, ws = gausshermite(K)
    N = K^2

    # Quadrature weights; exp(y²) undoes the Gauss-Hermite weight function
    weights = map(insertions) do g
        return [g(ys[α], ys[β]) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    end

    U, S, V = map(x -> convert(Array, x), svd_compact!(fmatrix_complex(ys, μ0, λ)))
    sqrtS = sqrt.(diag(S))

    arrays = map(_ -> zeros(T, N, N, N, N), insertions)
    perms = collect(permutations(1:4))

    @threads for i in 1:N
        for j in i:N, k in j:N, l in k:N
            factor = sqrtS[i] * sqrtS[j] * sqrtS[k] * sqrtS[l]
            sums = map(w -> zero(promote_type(eltype(w), Float64)), weights)
            for α in 1:K, β in 1:K
                idx = (α - 1) * K + β
                x = factor * U[idx, i] * U[idx, j] * V[k, idx] * V[l, idx]
                sums = map((s, w) -> s + w[α, β] * x, sums, weights)
            end

            # Fill all 24 symmetric permutations
            idxs = (i, j, k, l)
            for p in perms
                I = ntuple(n -> idxs[p[n]], 4)
                foreach((A, s) -> A[I...] = s, arrays, sums)
            end
        end
    end

    return map(A -> TensorMap(A, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N), arrays)
end

#####################################
#   Taylor expansion, ℤ₂ × ℤ₂       #
#####################################

# M[α + 1, β + 1] = ∫ d²ϕ (Re ϕ)^α (Im ϕ)^β exp(-(2 + μ0 / 2) |ϕ|² - λ / 4 |ϕ|⁴) for α, β = 0, …, N.
# Only even α and β give non-zero values.
function phi4_complex_z2z2_moments(N::Integer, μ0::Float64, λ::Float64)
    radial = radial_moments_complex(2N + 1, μ0, λ)
    M = zeros(Float64, N + 1, N + 1)
    for α in 0:2:N, β in α:2:N
        # Angular integral ∫₀^{2π} cos^α θ sin^β θ dθ times the radial integral ∫₀^∞ r^(α + β + 1) (...) dr
        val = 2 * beta((α + 1) / 2, (β + 1) / 2) * radial[α + β + 2]
        M[α + 1, β + 1] = val
        M[β + 1, α + 1] = val
    end
    return M
end

#####################################
#   Taylor expansion, U(1) and O(2) #
#####################################

# Bond space for the U(1) symmetric tensor: charge q appears K - |q| times.
function phi4_complex_space(::Type{U1Irrep}, K::Integer)
    return Vect[U1Irrep](q => K - abs(q) for q in (-(K - 1)):(K - 1))
end
# Bond space for the O(2) symmetric tensor: the charges ±q are combined into the
# two-dimensional irrep CU1Irrep(q, 2), which appears K - q times.
function phi4_complex_space(::Type{CU1Irrep}, K::Integer)
    return Vect[CU1Irrep]((0, 0) => K, ((q, 2) => K - q for q in 1:(K - 1))...)
end

# Exponents (n, n̄) of the monomial ϕⁿ ϕ̄ⁿ̄ belonging to the m-th copy of basis vector i of sector s.
# A basis vector with U(1) charge q has n - n̄ = q.
function phi4_complex_exponents(s::U1Irrep, ::Int, m::Int)
    q = Int(s.charge)
    return q >= 0 ? (m - 1 + q, m - 1) : (m - 1, m - 1 - q)
end
# For CU1Irrep(q, 2), the first basis vector carries U(1) charge +q, the second -q.
function phi4_complex_exponents(s::CU1Irrep, i::Int, m::Int)
    q = Int(s.j)
    return (q == 0 || i == 1) ? (m - 1 + q, m - 1) : (m - 1, m - 1 + q)
end

# Element of the Taylor-expanded tensor for legs with exponents (a, b), (c, d), (e, f), (g, h).
# It vanishes unless U(1) charge is conserved: (a - b) + (c - d) == (e - f) + (g - h).
function phi4_complex_taylor_element(moments, logfact, (a, b), (c, d), (e, f), (g, h))
    (a + c + f + h == b + d + e + g) || return 0.0
    n = a + b + c + d + e + f + g + h
    logdenom = 0.5 * (
        log(2) * n +
            logfact[a + 1] + logfact[b + 1] + logfact[c + 1] + logfact[d + 1] +
            logfact[e + 1] + logfact[f + 1] + logfact[g + 1] + logfact[h + 1]
    )
    return 2π * moments[n + 2] / exp(logdenom)
end

#####################################
#       TENSOR FUNCTIONS            #
#####################################

"""
    phi4_complex(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    phi4_complex(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    phi4_complex(::Type{Z2Irrep ⊠ Z2Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    phi4_complex(::Type{U1Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    phi4_complex(::Type{CU1Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)

Constructs the partition function tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

Compatible with no symmetry, explicit ℤ₂×ℤ₂ symmetry, explicit U(1) symmetry or explicit O(2) symmetry on each of its spaces.
Defaults to CU(1)/O(2) symmetry if the symmetry type is not provided.

# Arguments
- `K::Integer`: Approximation parameter.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

# Approximation parameter `K`
## Trivial (no symmetry)
The tensor is constructed by performing a [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature) to approximate the integrals in the partition function.
The bond dimension is equal to `K^2`.

## ℤ₂×ℤ₂ symmetry
The tensor is constructed by Taylor expanding the mixed sites term in the partition function.
The order of the Taylor expansion is `K`, which has to be even. The total bond dimension is `K^2`.

## U(1) symmetry
The tensor is constructed by Taylor expanding the mixed sites term in the partition function.
The order of the Taylor expansion is `K`, which has to be even. The total bond dimension is `K^2`.

Each basis state corresponds to a monomial ``ϕ^n \\bar{ϕ}^{\\bar{n}}`` with ``0 ≤ n, \\bar{n} < K`` and carries U(1) charge ``q = n - \\bar{n}``.
The charge `q` appears with multiplicity `K - |q|`.

## O(2) symmetry
Use `CU1Irrep` to make the full O(2) = U(1) ⋊ ℤ₂ symmetry of the model explicit, where the ℤ₂ is charge conjugation ``ϕ ↔ ϕ^\\dagger``.
The tensor is the same as the U(1) symmetric tensor, but expressed in a basis that also respects charge conjugation.
The order of the Taylor expansion is `K`, which has to be even. The total bond dimension is `K^2`.

The bond space consists of `CU1Irrep(0, 0)` with multiplicity `K` and the two-dimensional irreps `CU1Irrep(q, 2)`, which combine the U(1) charges ``±q``, with multiplicity `K - q` for `q = 1, …, K - 1`.

This is the default symmetry type if the symmetry type is not provided.

# Examples
```julia
    phi4_complex(CU1Irrep, 10, -1., 1.)
```

!!! info
    When studying this model with impurities, the tensor without symmetry should be constructed, as the impurity breaks the symmetry.

# References
Piceu Jarid and Adwait Naravane, but based on:
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)
* [Delcamp et. al. Phys. Rev. Research 2, 033278 (2020)](@cite delcamp2020)

See also: [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex(CU1Irrep, K, μ0, λ; kwargs...)
end
function phi4_complex(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    return only(phi4_complex_quadrature_tensors(K, μ0, λ, (PHI4_COMPLEX_INSERTIONS.one,); T))
end
function phi4_complex(::Type{Z2Irrep ⊠ Z2Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    iseven(K) || throw(ArgumentError("K must be even to split into even/odd groups"))

    # Re ϕ and Im ϕ each appear with a total power of at most 4(K - 1)
    moments = phi4_complex_z2z2_moments(4 * (K - 1), μ0, λ)
    logfact = logfactorial.(0:(K - 1))

    T_arr = zeros(T, K, K, K, K, K, K, K, K)

    @threads for a in 0:(K - 1)
        for c in 0:(K - 1), f in 0:(K - 1), h in 0:(K - 1)
            # Answer is zero if the total power of Re ϕ is odd
            α = a + c + f + h
            isodd(α) && continue

            for b in 0:(K - 1), d in 0:(K - 1), e in 0:(K - 1), g in 0:(K - 1)
                # Answer is zero if the total power of Im ϕ is odd
                β = b + d + e + g
                isodd(β) && continue

                logdenom = 0.5 * (
                    logfact[a + 1] + logfact[b + 1] + logfact[c + 1] + logfact[d + 1] +
                        logfact[e + 1] + logfact[f + 1] + logfact[g + 1] + logfact[h + 1]
                )
                T_arr[a + 1, b + 1, c + 1, d + 1, e + 1, f + 1, g + 1, h + 1] = moments[α + 1, β + 1] / exp(logdenom)
            end
        end
    end

    # Make it block diagonal: even powers first, then odd powers
    perm = vcat(1:2:K, 2:2:K)
    T_block = T_arr[perm, perm, perm, perm, perm, perm, perm, perm]

    # Build ℤ₂ × ℤ₂ spaces by fusing the Re ϕ and Im ϕ legs
    V = Z2Space([Z2Irrep(0) => K // 2, Z2Irrep(1) => K // 2])
    T_unfused = TensorMap(T_block, V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V)

    U = isometry(fuse(V, V), V ⊗ V)
    Udg = adjoint(U)

    @tensor T_fused[-1 -2; -3 -4] := T_unfused[1 2 3 4; 5 6 7 8] * U[-1; 1 2] * U[-2; 3 4] * Udg[5 6; -3] * Udg[7 8; -4]
    return T_fused
end
function phi4_complex(::Type{I}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64) where {I <: Union{U1Irrep, CU1Irrep}}
    iseven(K) || throw(ArgumentError("K must be even"))

    # ϕ and ϕ̄ each appear with a total power of at most 4(K - 1)
    moments = radial_moments_complex(8 * (K - 1) + 1, μ0, λ)
    logfact = logfactorial.(0:(K - 1))

    V = phi4_complex_space(I, K)
    t = zeros(T, V ⊗ V ← V ⊗ V)

    # Every block t[f₁, f₂] is the projection of the dense tensor D, written in the monomial basis, onto the fusion trees:
    # t[f₁, f₂] = 1 / dim(c) * Σ_μ C₁₂[i₁, i₂, μ] * D[i₁, i₂, i₃, i₄] * C₃₄[i₃, i₄, μ]
    # For U1Irrep all fusion tensors are trivial, and this reduces to copying the elements of D.
    @threads for (f₁, f₂) in collect(fusiontrees(t))
        s₁, s₂ = f₁.uncoupled
        s₃, s₄ = f₂.uncoupled
        c = f₁.coupled

        C₁₂ = fusiontensor(s₁, s₂, c)
        C₃₄ = fusiontensor(s₃, s₄, c)
        block = t[f₁, f₂]

        for i₁ in 1:dim(s₁), i₂ in 1:dim(s₂), i₃ in 1:dim(s₃), i₄ in 1:dim(s₄)
            w = sum(μ -> C₁₂[i₁, i₂, μ, 1] * C₃₄[i₃, i₄, μ, 1], 1:dim(c)) / dim(c)
            iszero(w) && continue

            for m₁ in axes(block, 1), m₂ in axes(block, 2), m₃ in axes(block, 3), m₄ in axes(block, 4)
                block[m₁, m₂, m₃, m₄] += w * phi4_complex_taylor_element(
                    moments, logfact,
                    phi4_complex_exponents(s₁, i₁, m₁), phi4_complex_exponents(s₂, i₂, m₂),
                    phi4_complex_exponents(s₃, i₃, m₃), phi4_complex_exponents(s₄, i₄, m₄),
                )
            end
        end
    end

    return t
end

"""
    phi4_complex_impϕ([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a ϕ operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

# Examples
```julia
    phi4_complex_impϕ(10, -1., 1.)
```

# References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕ(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕ(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕ(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    return only(phi4_complex_quadrature_tensors(K, μ0, λ, (PHI4_COMPLEX_INSERTIONS.ϕ,); T))
end

"""
    phi4_complex_impϕdag([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a ϕ† operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

# Examples
```julia
    phi4_complex_impϕdag(10, -1., 1.)
```

# References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕdag(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕdag(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕdag(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    return only(phi4_complex_quadrature_tensors(K, μ0, λ, (PHI4_COMPLEX_INSERTIONS.ϕdag,); T))
end

"""
    phi4_complex_impϕabs([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a √(ϕϕ†) operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

# Examples
```julia
    phi4_complex_impϕabs(10, -1., 1.)
```

# References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕabs(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕabs(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕabs(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    return only(phi4_complex_quadrature_tensors(K, μ0, λ, (PHI4_COMPLEX_INSERTIONS.ϕabs,); T))
end

"""
    phi4_complex_impϕ2([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a ϕϕ† operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

# Examples
```julia
    phi4_complex_impϕ2(10, -1., 1.)
```

# References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕ2(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕ2(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕ2(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    return only(phi4_complex_quadrature_tensors(K, μ0, λ, (PHI4_COMPLEX_INSERTIONS.ϕ2,); T))
end

"""
    phi4_complex_all([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs all the tensors: the partition function tensor and all the impurity tensors for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

It is faster to compute them all at once than one for one individually.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

# Returns
The tuple `(T, T_ϕ, T_ϕdag, T_ϕabs, T_ϕ2)`.

# Examples
```julia
    phi4_complex_all(10, -1., 1.)
```

# References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref).
"""
function phi4_complex_all(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_all(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_all(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    return phi4_complex_quadrature_tensors(K, μ0, λ, values(PHI4_COMPLEX_INSERTIONS); T)
end

"""
    phi4_complex_imp([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64, a::Integer, b::Integer; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a ``ϕ^a (ϕ^†)^b`` operator on this site, with ``ϕ = \\mathrm{Re}\\,ϕ + i\\,\\mathrm{Im}\\,ϕ``.
`(a, b) = (0, 0)` gives the pure tensor [`phi4_complex`](@ref) with `Trivial` symmetry, while `(1, 0)`, `(0, 1)` and `(1, 1)`
give respectively [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref) and [`phi4_complex_impϕ2`](@ref).
This makes it easy to construct all impurity tensors of order `(a, b)` for [`ImpurityHOTRG`](@ref), e.g.
`Dict((a, b) => phi4_complex_imp(K, μ0, λ, a, b) for a in 0:2, b in 0:2 if (a, b) != (0, 0))`.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

# Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.
- `a::Integer`: Power of ϕ of the impurity.
- `b::Integer`: Power of ϕ† of the impurity.

# Examples
```julia
    phi4_complex_imp(10, -1., 1., 2, 2) # (ϕϕ†)^2 impurity
```

# References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_imp(K::Integer, μ0::Float64, λ::Float64, a::Integer, b::Integer; kwargs...)
    return phi4_complex_imp(Trivial, K, μ0, λ, a, b; kwargs...)
end
function phi4_complex_imp(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, a::Integer, b::Integer; T::Type{<:Number} = ComplexF64)
    (a >= 0 && b >= 0) || throw(ArgumentError("The powers of the impurity must be non-negative"))
    # On-site insertion g(Re ϕ, Im ϕ) = ϕ^a (ϕ†)^b
    return only(phi4_complex_quadrature_tensors(K, μ0, λ, ((x, y) -> (x + y * im)^a * (x - y * im)^b,); T))
end
