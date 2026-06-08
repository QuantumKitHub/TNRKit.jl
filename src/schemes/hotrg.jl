"""
$(TYPEDEF)

Higher-Order Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T)

# Running the algorithm
    run!(::HOTRG, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

# References
* [Xie et. al. Phys. Rev. B 86 (2012)](@cite xie2012)

"""
mutable struct HOTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT

    function HOTRG(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T)
    end
end

#= 
NOTE: It's difficult to transpose the PFTensors planarly 
in order to reuse the y-compression code for x-compression.
Hence both are written explicitly.
=#

function _step_hotrg_y(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        Ux::AbstractTensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the y-direction
                    -3
                    |
            ┌---1---A2---3--┐
            |       |       |
        -1--Ux†      5      Ux-- -4
            |       |       |
            └---2---A1---4--┘
                    |
                    -2
    =#
    @tensor T[-1 -2; -3 -4] :=
        conj(Ux[1 2; -1]) * Ux[3 4; -4] * A2[1 5; -3 3] * A1[2 -2; 5 4]
    return T
end

function _step_hotrg_x(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        Uy::AbstractTensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the x-direction
                -3
                |
            ┌3--Uy-4┐
            |       |
        -1--A1--5---A2-- -4
            |       |
            └1-Uy†-2┘
                |
                -2
    =#
    @tensor T[-1 -2; -3 -4] :=
        A1[-1 1; 3 5] * A2[5 2; 4 -4] * conj(Uy[1 2; -2]) * Uy[3 4; -3]
    return T
end

function _get_hotrg_xproj(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        trunc::MatrixAlgebraKit.TruncationStrategy
    ) where {E, S}
    #= join in y-direction, keep x-indices open (A1 below A2)
    M M†                        M† M
            ┌---1---┐                   ┌---1---┐
            ↓       ↑                   ↑       ↓
    -1 -←--A2-←-2--A2†-←- -3    -1 -←--A2†--2-←-A2-←- -3
            ↓       ↑                   ↑       ↓
            5       6                   5       6
            ↓       ↑                   ↑       ↓
    -2 -←--A1-←-4--A1†-←- -4    -2 -←--A1†--4-←-A1-←- -4
            ↓       ↑                   ↑       ↓
            └---3---┘                   └---3---┘
    =#
    # get left unitary
    @plansor MM[-1 -2; -3 -4] :=
        A2[-1 5; 1 2] * A1[-2 3; 5 4] *
        conj(A2[-3 6; 1 2]) * conj(A1[-4 3; 6 4])
    # Project to hermitian because gross_neveu tests would not pass without
    _, U, ε = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    # get right unitary
    @plansor MM[-1 -2; -3 -4] :=
        conj(A2[2 5; 1 -1]) * conj(A1[4 3; 5 -2]) *
        A2[2 6; 1 -3] * A1[4 3; 6 -4]
    # Project to hermitian because gross_neveu tests would not pass without
    _, U′, ε′ = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    if ε > ε′
        U, ε = U′, ε′
    end
    return U, ε
end

function _get_hotrg_yproj(
        A1::TensorMap{E, S, 2, 2}, A2::TensorMap{E, S, 2, 2},
        trunc::MatrixAlgebraKit.TruncationStrategy
    ) where {E, S}
    #= join in x-direction, keep y-indices open (A1 on the left of A2)
    M M†                        M† M
            -3      -4              -3      -4
            ↓       ↓               ↓       ↓
        ┌-→-A1†--6-→A2†-→┐      ┌-←-A1-←-6--A2-←-┐
        ↑   ↓       ↓    ↓      ↑   ↓       ↓    ↓
        1   2       4    3      1   2       4    3
        ↑   ↓       ↓    ↓      ↑   ↓       ↓    ↓
        └-←-A1-←-5--A2-←-┘      └-→-A1†--5-→A2†-→┘
            ↓       ↓               ↓       ↓
            -1      -2              -1      -2
    =#
    # get bottom unitary
    @plansor MM[-1 -2; -3 -4] :=
        A1[1 -1; 2 5] * A2[5 -2; 4 3] *
        conj(A1[1 -3; 2 6]) * conj(A2[6 -4; 4 3])
    # Project to hermitian because gross_neveu tests would not pass without
    _, U, ε = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    # get top unitary
    @plansor MM[-1 -2; -3 -4] :=
        conj(A1[1 2; -1 5]) * conj(A2[5 4; -2 3]) *
        A1[1 2; -3 6] * A2[6 4; -4 3]
    # Project to hermitian because gross_neveu tests would not pass without
    _, U′, ε′ = eigh_trunc!(project_hermitian!(MM); trunc = trunc)

    if ε > ε′
        U, ε = U′, ε′
    end
    return U, ε
end

function step!(scheme::HOTRG, trunc::MatrixAlgebraKit.TruncationStrategy)
    Ux, = _get_hotrg_xproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_y(scheme.T, scheme.T, Ux)
    Uy, = _get_hotrg_yproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_x(scheme.T, scheme.T, Uy)
    return scheme
end

function Base.show(io::IO, scheme::HOTRG)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
