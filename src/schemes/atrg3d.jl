"""
$(TYPEDEF)

3D Anisotropic Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T)

# Running the algorithm
    run!(::ATRG_3D, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=defualt_Finalizer, finalize_beginning=true,verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

# References
* [Adachi et. al. Phys. Rev. B 102 (2020)](@cite adachi2020)
"""
mutable struct ATRG_3D{E, S, TT <: AbstractTensorMap{E, S, 2, 4}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT

    function ATRG_3D(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 4}}
        return new{E, S, TT}(T)
    end
end

function _step!(scheme::ATRG_3D, trunc::TruncationStrategy)
    U, S, V, _ = svd_trunc(permute(scheme.T, ((2, 5, 6), (3, 4, 1))); trunc = trunc)
    A = permute(U, ((4, 1), (2, 3)))
    D = permute(V, ((4, 1), (2, 3)))
    C = permute(U * S, ((4, 1), (2, 3)))
    B = permute(S * V, ((4, 1), (2, 3)))

    @tensor M[-1 -2; -3 -4 -5 -6] := B[1 -2; -3 -4] * C[-1 1; -5 -6]

    U, S, V, _ = svd_trunc(permute(M, ((2, 5, 6), (3, 4, 1))); trunc = trunc)
    sqrtS = sqrt(S)

    X = permute(U * sqrtS, ((4, 1), (2, 3)))
    Y = permute(sqrtS * V, ((4, 1), (2, 3)))

    @tensor AX[-1 -2; -3 -4 -5 -6] := A[1 -2; -3 -5] * X[-1 1; -4 -6]
    @tensor YD[-1 -2; -3 -4 -5 -6] := Y[1 -2; -3 -5] * D[-1 1; -4 -6]

    #The QR decompositions and construction of the four isometries
    _, R1 = left_orth(permute(YD, ((1, 2, 3, 4), (5, 6))))
    R2, _ = right_orth(permute(AX, ((5, 6), (1, 2, 3, 4))))
    _, R3 = left_orth(permute(YD, ((1, 2, 5, 6), (3, 4))))
    R4, _ = right_orth(permute(AX, ((3, 4), (1, 2, 5, 6))))

    @tensor temp1[-1; -2] := R1[-1; 1 2] * R2[1 2; -2]
    U1, S1, V1, _ = svd_trunc(temp1; trunc = trunc)
    inv_s1 = pseudopow(S1, -0.5)
    @tensor Proj_1[-1 -2; -3] := R2[-1 -2; 1] * adjoint(V1)[1; 2] * inv_s1[2; -3]
    @tensor Proj_2[-1; -2 -3] := inv_s1[-1; 1] * adjoint(U1)[1; 2] * R1[2; -2 -3]

    @tensor temp2[-1; -2] := R3[-1; 1 2] * R4[1 2; -2]
    U2, S2, V2, _ = svd_trunc(temp2; trunc = trunc)
    inv_s2 = pseudopow(S2, -0.5)
    @tensor Proj_3[-1 -2; -3] := R4[-1 -2; 1] * adjoint(V2)[1; 2] * inv_s2[2; -3]
    @tensor Proj_4[-1; -2 -3] := inv_s2[-1; 1] * adjoint(U2)[1; 2] * R3[2; -2 -3]

    @tensor H[-1 -2; -3 -4] := YD[-1 -2; 1 2 3 4] * Proj_3[1 2; -3] * Proj_1[3 4; -4]
    @tensor G[-1 -2; -3 -4] := AX[-1 -2; 1 2 3 4] * Proj_4[-3; 1 2] * Proj_2[-4; 3 4]

    @tensor scheme.T[-1 -2; -3 -4 -5 -6] := G[1 -2; -5 -6] * H[-1 1; -3 -4]
    return scheme
end

function step!(scheme::ATRG_3D, trunc::TruncationStrategy)
    _step!(scheme, trunc)

    scheme.T = permute(scheme.T, ((4, 6), (2, 5, 1, 3)))

    _step!(scheme, trunc)

    scheme.T = permute(scheme.T, ((4, 6), (2, 5, 1, 3)))

    _step!(scheme, trunc)

    return scheme.T = permute(scheme.T, ((4, 6), (2, 5, 1, 3)))
end

function Base.show(io::IO, scheme::ATRG_3D)
    println(io, "3D ATRG - Anisotropic TRG in 3D")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end

ATRG_3D_convcrit(steps::Int, data) = abs(log(data[end]) * 8.0^(1 - steps))
