"""
$(TYPEDEF)

Simple two-point correlation function for Higher-Order Tensor Renormalization Group

!!! info "Distance"
    Distance of `dist` is 2^{dist} sites apart. E.g. dist=3 means distance 2^3=8 sites apart.

# Constructors
    $(FUNCTIONNAME)(T, Timp1, Timp2, dist)

# Running the algorithm
    run!(scheme::CorrelationHOTRG, trunc::TruncationStrategy, niter::stopcrit[, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

# Fields

$(TYPEDFIELDS)

# References
Piceu J.
"""
mutable struct CorrelationHOTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
    "Pure Tensor"
    Tpure::TT

    "First type first order impurity tensor (Phase I)"
    Timp1::Union{TT, Nothing}

    "Second type first order impurity tensor (Phase I)"
    Timp2::Union{TT, Nothing}

    "The final impurity (Phase II & III)"
    Timp_final::Union{TT, Nothing}

    "Correlation distance (2^dist)"
    dist::Int

    "Iteration step"
    iter::Int

    function CorrelationHOTRG(
            Tpure::TT,
            Timp1::TT,
            Timp2::TT,
            dist::Int
        ) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}

        @assert dist ≥ 0 "Distance must be non-negative"

        @assert space(Tpure, 1) == space(Timp1, 1) == space(Timp2, 1)
        @assert space(Tpure, 2) == space(Timp1, 2) == space(Timp2, 2)
        @assert space(Tpure, 3) == space(Timp1, 3) == space(Timp2, 3)
        @assert space(Tpure, 4) == space(Timp1, 4) == space(Timp2, 4)

        return new{E, S, TT}(
            Tpure,
            Timp1,
            Timp2,
            nothing,    # no single impurity yet
            dist,
            0           # iteration starts at 0
        )
    end
end


"""
    step!(scheme::CorrelationHOTRG, trunc::TensorKit.TruncationStrategy)

Perform a single iteration step of the Correlation HOTRG algorithm.

This function progresses through three phases:
- **Phase 1**: Evolve the pure tensor with two separate impurity tensors.
- **Phase 2**: Merge the two impurity tensors into a single impurity tensor.
- **Phase 3**: Continue evolution with the single merged impurity tensor.

The algorithm determines which phase to execute based on the current iteration count.
After each phase, tensors are finalized and the iteration counter is incremented.

# Arguments
- `scheme::CorrelationHOTRG`: The HOTRG scheme containing the tensors and state.
- `trunc::TensorKit.TruncationStrategy`: The truncation scheme to apply during tensor operations.

# Returns
- `scheme::CorrelationHOTRG`: The updated scheme with evolved tensors and incremented iteration count.
"""
function step!(
        scheme::CorrelationHOTRG,
        trunc::TruncationStrategy
    )

    phase = _phase(scheme)

    if phase == 1
        # -----------------------------
        # PHASE 1 — TWO IMPURITIES
        # dist steps
        # -----------------------------

        phase1!(scheme, trunc)
        val = finalize_phase1!(scheme)

    elseif phase == 2
        # -----------------------------
        # PHASE 2 — MERGE IMPURITIES
        # 1 step
        # -----------------------------

        phase2!(scheme, trunc)
        val = finalize_phase23!(scheme)

        # Explicitly deactivate two-impurity tensors
        scheme.Timp1 = nothing
        scheme.Timp2 = nothing

    else
        # -----------------------------
        # PHASE 3 — SINGLE IMPURITY
        # (niter-dist-1) steps
        # -----------------------------

        phase3!(scheme, trunc)
        val = finalize_phase23!(scheme)
    end

    scheme.iter += 1
    return val
end


function run!(scheme::CorrelationHOTRG, trunc::TruncationStrategy, niter::stopcrit; verbosity = 1)
    # First check: assert realistic calculation
    @assert niter.n > scheme.dist "niter must be larger than dist"

    data = Vector()

    LoggingExtras.withlevel(; verbosity) do

        @infov 1 "Starting simulation\n $(scheme)\n"

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            val = step!(scheme, trunc)
            push!(data, val)
            steps += 1
            crit = niter(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(niter, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

function Base.show(io::IO, scheme::CorrelationHOTRG)
    println(io, "CorrelationHOTRG - Correlation Higher Order TRG")
    println(io, "  * Tpure: $(summary(scheme.Tpure))")
    println(io, "  * Timp1: $(summary(scheme.Timp1))")
    println(io, "  * Timp2: $(summary(scheme.Timp2))")
    println(io, "  * dist: $(scheme.dist)  → distance = $(2^scheme.dist)")
    return nothing
end

############################################################################
#                          HELPER FUNCTIONS                                #
############################################################################

function _phase(scheme::CorrelationHOTRG)
    if scheme.iter < scheme.dist        # dist steps
        return 1
    elseif scheme.iter == scheme.dist   # 1 step
        return 2
    else
        return 3                        # (niter-dist-1) steps
    end
end


function phase1!(scheme::CorrelationHOTRG, trunc::TruncationStrategy)
    Uy, _ = _get_hotrg_yproj(scheme.Tpure, scheme.Tpure, trunc)

    T = _step_hotrg_x(scheme.Tpure, scheme.Tpure, Uy)
    Timp1 = 0.5 * (_step_hotrg_x(scheme.Timp1, scheme.Tpure, Uy) + _step_hotrg_x(scheme.Tpure, scheme.Timp1, Uy))
    Timp2 = 0.5 * (_step_hotrg_x(scheme.Timp2, scheme.Tpure, Uy) + _step_hotrg_x(scheme.Tpure, scheme.Timp2, Uy))

    scheme.Tpure = T
    scheme.Timp1 = Timp1
    scheme.Timp2 = Timp2

    Ux, _ = _get_hotrg_xproj(scheme.Tpure, scheme.Tpure, trunc)

    T = _step_hotrg_y(scheme.Tpure, scheme.Tpure, Ux)
    Timp1 = 0.5 * (_step_hotrg_y(scheme.Timp1, scheme.Tpure, Ux) + _step_hotrg_y(scheme.Tpure, scheme.Timp1, Ux))
    Timp2 = 0.5 * (_step_hotrg_y(scheme.Timp2, scheme.Tpure, Ux) + _step_hotrg_y(scheme.Tpure, scheme.Timp2, Ux))

    scheme.Tpure = T
    scheme.Timp1 = Timp1
    scheme.Timp2 = Timp2

    return scheme
end

function phase2!(scheme::CorrelationHOTRG, trunc::TruncationStrategy)
    Uy, _ = _get_hotrg_yproj(scheme.Tpure, scheme.Tpure, trunc)

    T = _step_hotrg_x(scheme.Tpure, scheme.Tpure, Uy)
    T_imp = 0.5 * (_step_hotrg_x(scheme.Timp1, scheme.Timp2, Uy) + _step_hotrg_x(scheme.Timp2, scheme.Timp1, Uy))

    scheme.Tpure = T
    scheme.Timp_final = T_imp

    Ux, _ = _get_hotrg_xproj(scheme.Tpure, scheme.Tpure, trunc)

    T = _step_hotrg_y(scheme.Tpure, scheme.Tpure, Ux)
    T_imp = 0.5 * (_step_hotrg_y(scheme.Timp_final, scheme.Tpure, Ux) + _step_hotrg_y(scheme.Tpure, scheme.Timp_final, Ux))

    scheme.Tpure = T
    scheme.Timp_final = T_imp

    return scheme
end

function phase3!(scheme::CorrelationHOTRG, trunc::TruncationStrategy)
    Uy, _ = _get_hotrg_yproj(scheme.Tpure, scheme.Tpure, trunc)

    T = _step_hotrg_x(scheme.Tpure, scheme.Tpure, Uy)
    T_imp = 0.5 * (_step_hotrg_x(scheme.Timp_final, scheme.Tpure, Uy) + _step_hotrg_x(scheme.Tpure, scheme.Timp_final, Uy))

    scheme.Tpure = T
    scheme.Timp_final = T_imp

    Ux, _ = _get_hotrg_xproj(scheme.Tpure, scheme.Tpure, trunc)

    T = _step_hotrg_y(scheme.Tpure, scheme.Tpure, Ux)
    T_imp = 0.5 * (_step_hotrg_y(scheme.Timp_final, scheme.Tpure, Ux) + _step_hotrg_y(scheme.Tpure, scheme.Timp_final, Ux))

    scheme.Tpure = T
    scheme.Timp_final = T_imp

    return scheme
end
