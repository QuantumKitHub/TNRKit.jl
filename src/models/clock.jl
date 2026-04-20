function clock_tensor(q::Int, β::Real; T::Type{<:Number} = Float64)
    V = ℂ^q
    A_clock = zeros(T, V ⊗ V ← V ⊗ V)
    clock(i, j) = -cos(2π / q * (i - j))

    for i in 1:q, j in 1:q, k in 1:q, l in 1:q
        E = clock(i, j) + clock(j, l) + clock(l, k) + clock(k, i)
        A_clock[i, j, k, l] = exp(-β * E)
    end

    return A_clock
end

"""
    classical_clock(q::Int, β::Real; kwargs...)
    classical_clock(::Type{Trivial}, q::Int, β::Real; T::Type{<:Number} = Float64)
    classical_clock(::Type{ZNIrrep{N}}, q::Int, β::Real; T::Type{<:Number} = Float64) where {N}
    classical_clock(::Type{DNIrrep{N}}, q::Int, β::Real; T::Type{<:Number} = Float64) where {N}

Constructs the partition function tensor for the classical clock model with `q` states
and a given inverse temperature `β`.

Compatible with no symmetry, with explicit ℤq symmetry or Dq symmetry on each of its spaces.
Defaults to Dq symmetry if the symmetry type is not provided.
"""
function classical_clock(q::Int, β::Real; kwargs...)
    return classical_clock(DNIrrep{q}, q, β; kwargs...)
end
function classical_clock(::Type{Trivial}, q::Int, β::Real; kwargs...)
    return clock_tensor(q, β; kwargs...)
end
function classical_clock(::Type{ZNIrrep{N}}, q::Int, β::Real; T::Type{<:Number} = Float64) where {N}
    @assert N == q "number of irreps must match the number of states"
    A = classical_clock(Trivial, q, β; T = T)

    # Construct the Fourier matrix for the clock model
    Udat = zeros(ComplexF64, q, q)
    for i in 0:(q - 1)
        for j in 0:(q - 1)
            Udat[i + 1, j + 1] = cispi(2 / q * i * j) / sqrt(q)
        end
    end
    U = TensorMap(Udat, ℂ^q ← ℂ^q)

    @tensor Anew[-1 -2;-3 -4] := A[1 2; 3 4] * U[4; -4] * conj(U[1; -1]) * U[3; -3] * conj(U[2; -2])
    V = ZNSpace{q}(i => 1 for i in 0:(q - 1))
    t = TensorMap(convert(Array, Anew), V ⊗ V ← V ⊗ V)
    return T <: Real ? real(t) : t
end

function classical_clock(::Type{DNIrrep{N}}, q::Int, β::Real; T::Type{<:Number} = Float64) where {N}
    @assert N == q "number of irreps must match the number of states"

    FunZN, m = FunZN_Dihedral(q; T = T)

    bond = zeros(T, FunZN ← FunZN)

    for (s, f) in fusiontrees(bond)
        charge = f.coupled.j
        bond[s, f] .= sum(cos(2pi / q * spin * charge) * exp(β * cos(2pi / q * spin)) for spin in 0:(q - 1))
    end

    t = algebraic_initialization(m, bond)

    return t
end

function FunZN_Dihedral(N::Int; T::Type{<:Number} = Float64)
    n = N ÷ 2

    # Define which irreps are 1D irrep
    is_one_d(j) = iseven(N) ? (j == 0 || j == n) : (j == 0)

    FunZN = Rep[Dihedral{N}](DNIrrep{N}(k) => 1 for k in 0:n)

    m = zeros(T, FunZN ← FunZN ⊗ FunZN)

    for (s, f) in fusiontrees(m)
        upleft, upright = f.uncoupled
        down = f.coupled

        if is_one_d(upleft.j) || is_one_d(upright.j)
            m[s, f] .= 1
        elseif is_one_d(down.j)
            m[s, f] .= sqrt(2)
        else
            m[s, f] .= 1
        end
    end

    m /= sqrt(N)

    return FunZN, m
end
