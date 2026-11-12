abstract type stopcrit end

struct maxiter <: stopcrit
    n::Int
end

struct convcrit <: stopcrit
    Δ::Float64
    f::Function
end

struct MultipleCrit <: stopcrit
    crits::Vector{stopcrit}
end

Base.:&(a::stopcrit, b::stopcrit) = MultipleCrit([a, b])

(crit::maxiter)(steps::Int, data) = steps < crit.n
(crit::convcrit)(steps::Int, data) = crit.Δ < crit.f(steps, data)

# evaluate every criterion with short circuiting
(crit::MultipleCrit)(steps::Int, data) = !any(c -> !c(steps, data), crit.crits)

# information about which criterion is stopping the simulation
function stopping_info(crit::MultipleCrit, steps::Int, data)
    for c in crit.crits
        if !c(steps, data)
            return stopping_info(c, steps, data)
        end
    end
    return
end

function stopping_info(::maxiter, steps::Int, data)
    return "Maximum amount of iterations reached: $(steps)"
end

function stopping_info(crit::convcrit, steps::Int, data)
    return @sprintf "Convergence criterion reached: %.3e ≤ %.3e" crit.f(steps, data) crit.Δ
end

trivial_convcrit(Δ) = convcrit(Δ, (steps, data) -> last(data))
