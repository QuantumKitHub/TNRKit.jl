# Extra code to make output type available
function step! end
function finalize! end

"""
$(TYPEDEF)

Finalizer for TNR schemes

# Constructors
    Finalizer(f!::Function, E::Type)

A Finalizer holds a function `f!` that is to be applied to a TNR scheme after each step of the algorithm (and at the beginning if specified by `run!(;finalize_beginning=true)`, which is the default behavior).
The type parameter `E` indicates the output type of `f!`, which is used to create an array of the correct type to hold the outputs.
"""
struct Finalizer{E} # E is the output type of f
    f!::Function
end

function Finalizer(f::Function, E::Type)
    return Finalizer{E}(f)
end

output_type(finalizer::Finalizer{E}) where {E} = E

const default_Finalizer = Finalizer(finalize!, Float64)
const ImpurityTRG_Finalizer = Finalizer(finalize!, Tuple{Float64, Float64})
const ImpurityHOTRG_Finalizer = Finalizer(finalize!, Tuple{Float64, Float64, Float64, Float64})

# Finalization functions for the various TNR schemes
abstract type TNRScheme{E, S} end

function run!(scheme::TNRScheme, trscheme::TruncationStrategy, criterion::stopcrit, finalizer::Finalizer{E}; finalize_beginning = true, verbosity = 1) where {E}
    data = Vector{E}()

    LoggingExtras.withlevel(; verbosity) do

        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, finalizer.f!(scheme))
        end

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, trscheme)
            push!(data, finalizer.f!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

run!(scheme::TNRScheme, trscheme::TruncationStrategy, criterion::stopcrit; kwargs...) = run!(scheme, trscheme, criterion, default_Finalizer; kwargs...)
