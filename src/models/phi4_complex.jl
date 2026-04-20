#####################################
#       HELPER FUNCTIONS            #
#####################################

# For phi4_complex and such
function f_complex(ℝϕ1::Float64, ℂϕ1::Float64, ℝϕ2::Float64, ℂϕ2::Float64, μ0::Float64, λ::Float64)
    return exp(
        -1 / 2 * ((ℝϕ1 - ℝϕ2)^2 + (ℂϕ1 - ℂϕ2)^2)
            - μ0 / 8 * (ℝϕ1^2 + ℂϕ1^2 + ℝϕ2^2 + ℂϕ2^2)
            - λ / 16 * ((ℝϕ1^2 + ℂϕ1^2)^2 + (ℝϕ2^2 + ℂϕ2^2)^2)
    )
end

# For phi4_complex and such
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

# For phi4_complex_U1
function precompute_moments_complex(K, μ0, λ)
    a = 2 + μ0 / 2
    b = λ / 4     # convention, yeah, convention
    nmax = 8 * (K - 1) + 1
    M = zeros(Float64, nmax + 1)

    for n in 0:nmax
        f(r) = begin
            logval = n * log(r) - a * r^2 - b * r^4
            return exp(logval)        # safe everywhere, never NaN
        end

        val, _ = quadgk(f, 0.0, Inf; rtol = 1.0e-8, maxevals = 10^7)
        M[n + 1] = val
    end
    return M
end

# For phi4_complex_Z2Z2
function precompute_radial_integrals(N, μ0, λ; rtol = 1.0e-8)

    a = 2 + μ0 / 2
    b = λ / 4

    b >= 0 || error("Integral diverges for λ < 0")

    I = Dict{Int, Float64}()

    # Only even n up to 2N are needed
    for n in 0:2:2N

        f(r) = r^(n + 1) * exp(-a * r^2 - b * r^4)

        val, _ = quadgk(f, 0, Inf; rtol = rtol)

        I[n] = val
    end

    return I
end

# For phi4_complex_Z2Z2
function moment_matrix(N, μ0, λ; rtol = 1.0e-8)

    M = zeros(Float64, N + 1, N + 1)

    # Precompute radial integrals
    I = precompute_radial_integrals(N, μ0, λ; rtol = rtol)

    for α in 0:N
        for β in α:N   # upper triangle only

            if iseven(α) && iseven(β)

                n = α + β

                C = 2 * beta((α + 1) / 2, (β + 1) / 2)

                val = C * I[n]

                M[α + 1, β + 1] = val
                M[β + 1, α + 1] = val  # symmetry
            end
        end
    end

    return M
end

function phi4_complex_tensor(f::TensorMap{TT, SS, NN, NN}, weights::Matrix{<:Number}; T::Type{<:Number} = ComplexF64) where {TT, SS, NN} # for Trivial symmetry
    K = size(weights, 1)
    N = K^2
    perms = collect(permutations(1:4))  # 24 total

    U, S, V = svd_compact!(f)
    T_arr = zeros(T, N, N, N, N)

    @threads for i in 1:N
        for j in i:N, k in j:N, l in k:N
            s = 0.0
            factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
            for α in 1:K, β in 1:K
                s += factor *
                    weights[α, β] *
                    U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] *
                    V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
            end

            # Fill all 24 symmetric permutations
            idxs = (i, j, k, l)
            for p in perms
                ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                T_arr[ii, jj, kk, ll] = s
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end


#####################################
#       TENSOR FUNCTIONS            #
#####################################

"""
    phi4_complex(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    phi4_complex(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    phi4_complex(::Type{Z2Irrep ⊠ Z2Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    phi4_complex(::Type{U1Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)

Constructs the partition function tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

Compatible with no symmetry, explicit ℤ₂×ℤ₂ symmetry or explicit U(1) symmetry on each of its spaces.
Defaults to U(1) symmetry if the symmetry type is not provided.

### Arguments
- `K::Integer`: Approximation parameter.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

### Approximation parameter `K`
#### Trivial (no symmetry)
The tensor is constructed by performing a Gauss-Hermite quadrature to approximate the integrals in the partition function.
The bond dimension is equal to `K^2`.

#### ℤ₂×ℤ₂ symmetry
The tensor is constructed by Taylor expanding the mixed sites term in the partition function.
The order of the Taylor expansion is `K`. The total bond dimension is `K^2`.

#### U(1) symmetry
The tensor is constructed by Taylor expanding the mixed sites term in the partition function.
The order of the Taylor expansion is `K`. The total bond dimension is `K^2`.

### Examples
```julia
    phi4_complex(10, -1., 1.)
```

!!! info
    When studying this model with impurities, the tensor without symmetry should be constructed, as the impurity breaks the symmetry.

### References
Piceu Jarid and Adwait Naravane, but based on:
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)
* [Delcamp et. al. Phys. Rev. Research 2, 033278 (2020)](@cite delcamp2020)

See also: [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex(U1Irrep, K, μ0, λ; kwargs...)
end
function phi4_complex(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # Determine weights
    weights = [ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    # Build tensor
    t = phi4_complex_tensor(f, weights; T = T)
    return t
end
function phi4_complex(::Type{Z2Irrep ⊠ Z2Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    if K % 2 != 0
        error("K must be even to split into even/odd groups")
    end

    # precompute moment
    moments = moment_matrix(4 * K, μ0, λ)
    # log factorials 0..K-1
    logfact = log.(factorial.(0:(K - 1)))


    T_arr = zeros(T, K, K, K, K, K, K, K, K)

    @threads for a in 0:(K - 1)
        for c in 0:(K - 1), f in 0:(K - 1), h in 0:(K - 1)
            # Answer is zero if a+c+f+h is odd
            if isodd(a + c + f + h)
                continue
            end

            for b in 0:(K - 1), d in 0:(K - 1), e in 0:(K - 1), g in 0:(K - 1)
                # Answer is zero if b+d+e+g is odd
                if isodd(b + d + e + g)
                    continue
                end

                # Calculate moment
                α = a + c + f + h
                β = b + d + e + g
                M = moments[α + 1, β + 1]

                # denomenator via logfacts
                logdenom = 0.5 * (logfact[a + 1] + logfact[b + 1] + logfact[c + 1] + logfact[d + 1] + logfact[e + 1] + logfact[f + 1] + logfact[g + 1] + logfact[h + 1])
                denom = exp(logdenom)

                val = M / denom

                # store into array (indices +1)
                T_arr[a + 1, b + 1, c + 1, d + 1, e + 1, f + 1, g + 1, h + 1] = val
            end
        end
    end

    # Make it block diagonal
    evens = 1:2:K
    odds = 2:2:K
    perm = vcat(evens, odds)
    T_block = T_arr[perm, perm, perm, perm, perm, perm, perm, perm]


    # Build Z2 spaces
    V = Z2Space([Z2Irrep(0) => K // 2, Z2Irrep(1) => K // 2])
    T_unfused = TensorMap(T_block, V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V)

    U = isometry(fuse(V, V), V ⊗ V)
    Udg = adjoint(U)

    @tensor T_fused[-1 -2; -3 -4] := T_unfused[1 2 3 4; 5 6 7 8] * U[-1; 1 2] * U[-2; 3 4] * Udg[5 6; -3] * Udg[7 8; -4]
    return T_fused
end
function phi4_complex(::Type{U1Irrep}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = Float64)
    if K % 2 != 0
        error("K must be even")
    end

    moments = precompute_moments_complex(K, μ0, λ)
    logfact = log.(factorial.(0:(K - 1)))

    V1 = U1Space([U1Irrep(q) => 1 for q in 0:(K - 1)]...)
    V2 = U1Space([U1Irrep(q) => 1 for q in 0:-1:(-K + 1)]...)
    W = fuse(V1, V2)

    # Build multiplicity index lookup:
    # For fused charge q = a - B, the pairs (a, B) are ordered by increasing a.
    # mult_index[q][a] = the 1-based index into the W-block for that pair.
    mult_index = Dict{Int, Dict{Int, Int}}()
    for q in (-(K - 1)):(K - 1)
        pairs = Int[]
        for a in max(0, q):(min(K - 1, q + K - 1))
            B = a - q
            (0 <= B <= K - 1) || continue
            push!(pairs, a)
        end
        mult_index[q] = Dict(a => i for (i, a) in enumerate(pairs))
    end

    T_fused = TensorMap(zeros(T, K^2, K^2, K^2, K^2), W ⊗ W ← W ⊗ W)

    for (split_tree, fuse_tree) in fusiontrees(T_fused)
        q1 = Int(split_tree.uncoupled[1].charge)
        q2 = Int(split_tree.uncoupled[2].charge)
        q3 = Int(fuse_tree.uncoupled[1].charge)
        q4 = Int(fuse_tree.uncoupled[2].charge)

        block = T_fused[split_tree, fuse_tree]
        # block has shape (mult(q1) × mult(q2)) × (mult(q3) × mult(q4))
        # index as block[i1, i2, i3, i4] where each i is the multiplicity index

        for a in max(0, q1):min(K - 1, q1 + K - 1)
            B = a - q1;  (0 <= B <= K - 1) || continue
            i1 = mult_index[q1][a]

            for c in max(0, q2):min(K - 1, q2 + K - 1)
                D = c - q2;  (0 <= D <= K - 1) || continue
                i2 = mult_index[q2][c]

                for e in max(0, q3):min(K - 1, q3 + K - 1)
                    F = e - q3;  (0 <= F <= K - 1) || continue
                    i3 = mult_index[q3][e]

                    for g in max(0, q4):min(K - 1, q4 + K - 1)
                        H = g - q4;  (0 <= H <= K - 1) || continue
                        i4 = mult_index[q4][g]

                        sum_power = a + B + c + D + e + F + g + H
                        M = moments[sum_power + 2]
                        (M == 0.0) && continue

                        logdenom = 0.5 * (
                            log(2) * sum_power +
                                logfact[a + 1] + logfact[B + 1] + logfact[c + 1] + logfact[D + 1] +
                                logfact[e + 1] + logfact[F + 1] + logfact[g + 1] + logfact[H + 1]
                        )

                        block[i1, i2, i3, i4] += 2π * M / exp(logdenom)
                    end
                end
            end
        end
    end

    return T_fused
end

"""
    phi4_complex_impϕ([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a ϕ operator on this site.
    
It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕ(10, -1., 1.)
```

### References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕ(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕ(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕ(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # Determine weights
    weights = [(ys[α] + ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    # Build tensor
    T = phi4_complex_tensor(f, weights; T = T)
    return T
end


"""
    phi4_complex_impϕdag([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a ϕ† operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕdag(10, -1., 1.)
```

### References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕdag(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕdag(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕdag(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # Determine weights
    weights = [(ys[α] - ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    # Build tensor
    T = phi4_complex_tensor(f, weights; T = T)
    return T
end

"""
    phi4_complex_impϕabs([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a √(ϕϕ†) operator on this site.
    
It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕabs(10, -1., 1.)
```

### References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕabs(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕabs(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕabs(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # Determine weights
    weights = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    # Build tensor
    T = phi4_complex_tensor(f, weights; T = T)
    return T
end

"""
    phi4_complex_impϕ2([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs the impurity tensor for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

The impurity is a ϕϕ† operator on this site.
    
It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕ2(10, -1., 1.)
```

### References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_impϕ2(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_impϕ2(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_impϕ2(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # Determine weights
    weights = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    # Build tensor
    T = phi4_complex_tensor(f, weights; T = T)
    return T
end

"""
    phi4_complex_all([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)

Constructs all the tensors: the partition function tensor and all the impurity tensors for a 2D square lattice for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0` and interaction constant `λ`.

It is faster to compute them all at once than one for one individually.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.

### Examples
```julia
    phi4_complex_all(10, -1., 1.)
```

### References
Piceu Jarid, but based on [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref).
"""
function phi4_complex_all(K::Integer, μ0::Float64, λ::Float64; kwargs...)
    return phi4_complex_all(Trivial, K, μ0, λ; kwargs...)
end
function phi4_complex_all(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64; T::Type{<:Number} = ComplexF64)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = svd_compact!(f)

    N = K^2

    T_arr = zeros(T, N, N, N, N)
    T_ϕ_arr = zeros(T, N, N, N, N)
    T_ϕdag_arr = zeros(T, N, N, N, N)
    T_ϕabs_arr = zeros(T, N, N, N, N)
    T_ϕ2_arr = zeros(T, N, N, N, N)

    w = [ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕ = [(ys[α] + ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕdag = [(ys[α] - ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕabs = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕ2 = [(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]


    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N, k in j:N, l in k:N
            s = 0.0
            s_ϕ = 0.0
            s_ϕdag = 0.0
            s_ϕabs = 0.0
            s_ϕ2 = 0.0
            factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
            for α in 1:K, β in 1:K
                s += factor * w[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                s_ϕ += factor * w_ϕ[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                s_ϕdag += factor * w_ϕdag[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                s_ϕabs += factor * w_ϕabs[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                s_ϕ2 += factor * w_ϕ2[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
            end

            # Fill all 24 symmetric permutations
            idxs = (i, j, k, l)
            for p in perms
                ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                T_arr[ii, jj, kk, ll] = s
                T_ϕ_arr[ii, jj, kk, ll] = s_ϕ
                T_ϕdag_arr[ii, jj, kk, ll] = s_ϕdag
                T_ϕabs_arr[ii, jj, kk, ll] = s_ϕabs
                T_ϕ2_arr[ii, jj, kk, ll] = s_ϕ2
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕ = TensorMap(T_ϕ_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕdag = TensorMap(T_ϕdag_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕabs = TensorMap(T_ϕabs_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕ2 = TensorMap(T_ϕ2_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T, T_ϕ, T_ϕdag, T_ϕabs, T_ϕ2
end
