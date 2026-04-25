"""
$(TYPEDEF)

Periodic Transfer Matrix Renormalization Group 

### Constructors
    $(FUNCTIONNAME)(T)

### Running the algorithm
    run!(::PTMRG, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

# TODO: add the proper scaling factor here
Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Fedorovich et. al. Phys. Rev. B 111 (2025)](@cite fedorovich2025)

"""
mutable struct PTMRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT
    C::TT
    h::TT
    v::TT

    function PTMRG(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T, copy(T), copy(T), copy(T))
    end
end

function step!(scheme::PTMRG, trunc::MatrixAlgebraKit.TruncationStrategy)
    Ux, = _get_hotrg_xproj(scheme.h, scheme.C, trunc)
    scheme.C = _step_hotrg_y(scheme.h, scheme.C, Ux)

    Ux, = _get_hotrg_xproj(scheme.T, scheme.v, trunc)
    scheme.v = _step_hotrg_y(scheme.T, scheme.v, Ux)

    Uy, = _get_hotrg_yproj(scheme.C, scheme.v, trunc)
    scheme.C = _step_hotrg_x(scheme.C, scheme.v, Uy)

    Uy, = _get_hotrg_yproj(scheme.h, scheme.T, trunc)
    scheme.h = _step_hotrg_x(scheme.h, scheme.T, Uy)
    return scheme
end

function Base.show(io::IO, scheme::PTMRG)
    println(io, "PTMRG - Periodic Tranfer Matrix Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * h: $(summary(scheme.h))")
    println(io, "  * v: $(summary(scheme.v))")
    return nothing
end
