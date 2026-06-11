"""
Construct partition function tensor from nearest neighbor
Trotter gate of (1 + 1)D quantum models
```
                                                2       3
                                                  ↘   ↙
                                                    S4
    3      4        2                  3            ↓
      ↘︎  ↙︎            ↘              ↙              1
      gate      =      S1 ← 3   1 ← S3     or
      ↙︎  ↘︎            ↙              ↘              3
    1      2        1                  2            ↓
                                                    S2
                                                  ↙   ↘
                                                1       2
```
The partition function tensor is
```
                3'
                ↓
                S2
              ↙   ↘
            b       c                   3'
          ↙           ↘                 ↓
    1'← S3            S1 ← 4' --->  1'← T ← 4'
          ↘           ↙                 ↓
            a       d                   2'
              ↘   ↙
                S4
                ↓
                2'
```
"""
function gate_to_tensor(
        gate::AbstractTensorMap{E, S, 2, 2}; trunc = truncerror(; rtol = 1.0e-8)
    ) where {E, S}
    s1, s3 = SVD12(permute(gate, ((1, 3), (2, 4))), trunc)
    s2, s4 = SVD12(gate, trunc)
    @tensor T[-1 -2; -3 -4] := s3[-1 a b] * s4[-2 a d] * s2[b c -3] * s1[d c -4]
    return T
end

"""
Exponentially stack `2^nfold` copies of `T` in vertical direction using HOTRG.
"""
function vertical_stack_exp(
        T::AbstractTensorMap{E, S, 2, 2}, nfold::Integer, trunc::TruncationStrategy
    ) where {E, S}
    @assert nfold >= 1
    T2 = copy(T)
    for _ in 1:nfold
        Ux, = _get_hotrg_xproj(T2, T2, trunc)
        T2 = _step_hotrg_y(T2, T2, Ux)
    end
    return T2
end

"""
Linearly stack `n` copies of `T` in vertical direction using HOTRG.
"""
function vertical_stack_linear(
        T::AbstractTensorMap{E, S, 2, 2}, n::Integer, trunc::TruncationStrategy
    ) where {E, S}
    @assert n >= 1
    T2 = copy(T)
    for _ in 1:(n - 1)
        Ux, = _get_hotrg_xproj(T, T2, trunc)
        T2 = _step_hotrg_y(T, T2, Ux)
    end
    return T2
end

# Nearest neighbor (1+1)D quantum Hamiltonian gates
# =================================================

"""
Partition function tensor for 1D transverse field Ising chain
```
    H(PBC) = -J ∑_i (σz_i σz_{i+1} + g σx_i)
```
Allowed `symm`: Trivial, Z2Irrep.
"""
function ising_chain(
        elt::Type{<:Number}, symm::Type{<:Sector}, dt::Float64;
        J::Float64 = 1.0, g::Float64 = 0.0
    )
    ZZ = 4 * SO.S_z_S_z(elt, symm)
    X = SO.σˣ(elt, symm)
    unit = TensorKit.id(codomain(X))
    gate = ZZ + (g / 2) * (X ⊗ unit + unit ⊗ X)
    gate = exp(dt * J * gate)
    return gate_to_tensor(gate)
end
ising_chain(elt::Type{<:Number}, ϵ::Float64; kwargs...) =
    ising_chain(elt, Trivial, ϵ; kwargs...)
ising_chain(symm::Type{<:Sector}, ϵ::Float64; kwargs...) =
    ising_chain(ComplexF64, symm, ϵ; kwargs...)
