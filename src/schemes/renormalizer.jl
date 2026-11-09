abstract type TNRAlgorithm end

"""
$(TYPEDEF)

Iterator that performs RG coarse-graining steps on network tensors.

# Fields
$(TYPEDFIELDS)

# Iterator
In a for-loop, each iteration yields `(state, data)` after each RG step.
Stops when the algorithm's stopping criterion is met.
"""
mutable struct Renormalizer{A <: TNRAlgorithm, S, D}
    "Algorithm configuration (truncation, maxiter, etc.)"
    alg::A
    "Algorithm-specific state holding all involved tensors"
    state::S
    "Accumulated per-step data"
    data::Vector{D}
    "Number of RG steps performed so far"
    step::Int
end

Renormalizer(alg::A, state::S, data::Vector{D}) where {A, S, D} = Renormalizer{A, S, D}(alg, state, data, 0)
Renormalizer(alg::A, state::S, data::Vector{D}, step::Int) where {A, S, D} = Renormalizer{A, S, D}(alg, state, data, step)

function Base.iterate(r::Renormalizer)
    return iterate(r, 0)
end

function Base.iterate(r::Renormalizer, n::Int)
    r.alg.stop(n, r.data) || return nothing
    step!(r.state, r.alg)
    val = finalize!(r.state)
    push!(r.data, val)
    r.step = n + 1
    return ((r.state, r.data), n + 1)
end

function Base.show(io::IO, r::Renormalizer)
    println(io, "Renormalizer")
    println(io, "  * algorithm: $(nameof(typeof(r.alg)))")
    println(io, "  * state: $(nameof(typeof(r.state)))")
    println(io, "  * step: $(r.step)")
    println(io, "  * data: $(length(r.data)) entries")
    return nothing
end

"""
    rgstep!(r::Renormalizer)

Perform one RG coarse-graining step. Wraps `Base.iterate`.
Throws an error if the stopping criterion has already been met.
"""
function rgstep!(r::Renormalizer)
    r.alg.stop(r.step, r.data) || error("stop criterion reached")
    iterate(r, r.step)
    return r.state, r.data
end

"""
    run!(r::Renormalizer; verbosity=1)

Run the RG flow to completion. Returns `(final_state, data)`.
"""
function run!(r::Renormalizer; verbosity = 1)
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(r.state)\n"
        t = @elapsed for (_, data) in r
            @infov 2 "Step $(r.step), data[end]: $(data[end])"
        end
        @infov 1 "Simulation finished\n $(stopping_info(r.alg.stop, r.step, r.data))\n Elapsed time: $(t)s\n Iterations: $(r.step)"
    end
    return r.state, r.data
end
