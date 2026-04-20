"""
$(TYPEDEF)

Loop Optimization for Tensor Network Renormalization

### Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(TA, TB)
    $(FUNCTIONNAME)(unitcell_2x2::Matrix{T})

### Running the algorithm
    run!(::LoopTNR, trunc::TruncationStrategy, criterion::stopcrit, parameters::LoopParameters, finalizer::Finalizer[,
              entanglement_criterion::stopcrit, finalize_beginning=true, verbosity=1])
    
    run!(::LoopTNR, trscheme::TruncationStrategy, criterion::stopcrit, parameters::LoopParameters; kwargs...)

    run!(::LoopTNR, trscheme::TruncationStrategy, criterion::stopcrit[finalize_beginning=true, verbosity=1])

### LoopParameters
See also: [`LoopParameters`](@ref)
This stuct is used to set all internal parameters in LoopTNR.
It can also be used to control whether Krylov methods are used (default: false)
And whether nuclear norm regularization is used (default: false)

### Fields

$(TYPEDFIELDS)

### References
* [Yang et. al. Phys. Rev. Letters 118 (2017)](@cite yang2017)
* [Homma et. al. Phys. Rev. Res. 6 (2024)](@cite homma2024a)

"""
mutable struct LoopTNR{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "Central tensor on sublattice A"
    TA::TT

    "Central tensor on sublattice B"
    TB::TT

    function LoopTNR(TA::TT, TB::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(TA, TB)
    end
    function LoopTNR(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T, copy(T))
    end
end

"""
    $(TYPEDEF)

Parameters used during LoopTNR.
This struct allows the user to control how the linear problem is solved.
It also allows the user to turn on nuclear norm regularization.

### Fields

    $(TYPEDFIELDS)
"""
@kwdef struct LoopParameters{A}
    sweeping::stopcrit = maxiter(20) & convcrit(1.0e-9, (steps, cost) -> abs(cost[end]))
    one_loop_init::Bool = true
    truncentanglement::TruncationStrategy = trunctol(; rtol = 1.0e-14)

    # Krylov parameters
    "Use Krylov methods to solve the linear system in loop optimization. Default = false, which uses the backslash operator."
    krylov::Bool = false
    "Default Krylov algorithm is GMRES with maxiter = 250, krylovdim = 128, tol = 1.0e-10, verbosity = 0."
    krylovalg::A = GMRES(; maxiter = 250, krylovdim = 128, tol = 1.0e-10, verbosity = 0)

    # NNR parameters
    "Use Nuclear Norm Regularisation. Default = false"
    nuclear_norm::Bool = false
    ρ::Float64 = 0.8
    ξ_init::Float64 = 1.0e-5
    ξ_min::Float64 = 1.0e-7
end

"""
    LoopTNR(
        unitcell_2x2::Matrix{T},
        trunc::TruncationStrategy,
        loop_condition::LoopParameters
    ) where {T <: AbstractTensorMap{<:Any, <:Any, 2, 2}}

Initialize LoopTNR using a network with 2 x 2 unit cell, 
by first performing one round of loop optimization to reduce
the network to a bipartite one (without normalization). 
"""
function LoopTNR(
        unitcell_2x2::Matrix{T};
        trunc::TruncationStrategy,
        loop_condition::LoopParameters
    ) where {T <: AbstractTensorMap{<:Number, <:VectorSpace, 2, 2}}
    ψA = Ψ_A(unitcell_2x2)
    ψB = loop_opt(ψA, trunc, loop_condition, 0)
    TA, TB = ΨB_to_TATB(ψB)
    return LoopTNR(TA, TB)
end

function _check_dual(T::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    return [isdual(space(T, ax)) for ax in 1:4] == [0, 0, 1, 1]
end

# Function to initialize the list of tensors Ψ_A, making it an MPS on a ring
function Ψ_A(unitcell_2x2::Matrix{<:AbstractTensorMap{E, S, 2, 2}}) where {E, S}
    size(unitcell_2x2) == (2, 2) || error("Input unit cell must have 2 x 2 size.")
    ΨA = [
        transpose(unitcell_2x2[1, 1], ((2,), (1, 3, 4)); copy = true),
        transpose(unitcell_2x2[1, 2], ((1,), (3, 4, 2)); copy = true),
        transpose(unitcell_2x2[2, 2], ((3,), (4, 2, 1)); copy = true),
        transpose(unitcell_2x2[2, 1], ((4,), (2, 1, 3)); copy = true),
    ]
    return ΨA
end
function Ψ_A(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    ΨA = [
        transpose(TA, ((2,), (1, 3, 4)); copy = true),
        transpose(TB, ((1,), (3, 4, 2)); copy = true),
        transpose(TA, ((3,), (4, 2, 1)); copy = true),
        transpose(TB, ((4,), (2, 1, 3)); copy = true),
    ]
    return ΨA
end
function Ψ_A(scheme::LoopTNR)
    return Ψ_A(scheme.TA, scheme.TB)
end

# Function to construct MPS Ψ_B from MPS Ψ_A. Using a large cut-off dimension in SVD but a small cut-off dimension in loop to increase the precision of initialization.
function Ψ_B(ΨA::Vector{<:AbstractTensorMap{E, S, 1, 3}}, trunc::TruncationStrategy, loop_condition::LoopParameters) where {E, S}
    @assert trunc isa MatrixAlgebraKit.TruncationByOrder
    NA = length(ΨA)

    loop_condition.one_loop_init ? _trunc = truncrank(trunc.howmany * 2) : _trunc = trunc
    #= 
            |     |
            2 --- 3
          ↗         ↘
    --- 1             4 ---
        |             |
    --- 8             5 ---
          ↘         ↗
            7 --- 6
            |     |
    =#
    ΨB = [
        collect(SVD12(ΨA[1], _trunc; reversed = true));
        collect(SVD12(ΨA[2], _trunc; reversed = true));
        collect(SVD12(ΨA[3], _trunc));
        collect(SVD12(ΨA[4], _trunc));
    ]

    if loop_condition.one_loop_init
        ΨB_function(steps, data) = abs(data[end])
        criterion = maxiter(10) & convcrit(1.0e-12, ΨB_function)

        in_inds = ones(Int, 2 * NA)
        out_inds = 2 * ones(Int, 2 * NA)

        PR_list, PL_list = find_projectors(ΨB, in_inds, out_inds, criterion, trunc & loop_condition.truncentanglement)
        MPO_disentangled!(ΨB, in_inds, out_inds, PR_list, PL_list)
    end

    return ΨB
end

# Construct the list of transfer matrices for ΨAΨA
# ---1'--A--3'---
#       | |
#       1 2
#       | |
# ---2'--A--4'---
function ΨAΨA(ΨA::Vector{<:AbstractTensorMap{E, S, 1, 3}}) where {E, S}
    return map(ΨA) do A
        return @plansor AA[-1 -2; -3 -4] := A[-2; 1 2 -4] * conj(A[-1; 1 2 -3])
    end
end

# Construct the list of transfer matrices for ΨBΨB
# ---1'--B--3'---
#        |
#        1
#        |
# ---2'--B--4'---
function ΨBΨB(ΨB::Vector{<:AbstractTensorMap{E, S, 1, 2}}) where {E, S}
    return map(ΨB) do B
        return @plansor BB[-1 -2; -3 -4] := B[-2; 1 -4] * conj(B[-1; 1 -3])
    end
end

# Construct the list of transfer matrices for ΨBΨA
# ---1'--B-3-B--3'---
#        |   |
#        1   2
#         | |
# ---2'----A----4'---
function ΨBΨA(ΨB::Vector{<:AbstractTensorMap{E, S, 1, 2}}, ΨA::Vector{<:AbstractTensorMap{E, S, 1, 3}}) where {E, S}
    @assert length(ΨB) == 2 * length(ΨA)
    return map(eachindex(ΨA)) do i
        return @plansor temp[-1 -2; -3 -4] := conj(ΨB[2 * i - 1][-1; 1 3]) *
            ΨA[i][-2; 1 2 -4] * conj(ΨB[2 * i][3; 2 -3])
    end
end

#Entanglement Filtering
entanglement_function(steps, data) = abs(data[end])
default_entanglement_criterion = maxiter(100) & convcrit(1.0e-15, entanglement_function)

function _entanglement_filtering(
        TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2},
        entanglement_criterion::stopcrit, trunc::TruncationStrategy
    ) where {E, S}
    ΨA = Ψ_A(TA, TB)
    PRs, PLs = find_projectors(
        ΨA, [1, 1, 1, 1], [3, 3, 3, 3],
        entanglement_criterion, trunc
    )
    @plansor TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PRs[4][1; -1] * PLs[1][-2; 2] * PRs[2][4; -4] * PLs[3][-3; 3]
    @plansor TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PLs[2][-1; 1] * PRs[3][2; -2] * PLs[4][-4; 4] * PRs[1][3; -3]
    @assert _check_dual(TA) && _check_dual(TB)
    return TA, TB
end

# Entanglement filtering function
function entanglement_filtering!(
        scheme::LoopTNR,
        trunc::TruncationStrategy,
        entanglement_criterion::stopcrit = default_entanglement_criterion
    )
    scheme.TA, scheme.TB = _entanglement_filtering(
        scheme.TA, scheme.TB, entanglement_criterion, trunc
    )
    return scheme
end

# Optimisation functions

# Function to compute the half of the matrix N by inputting the left and right SS transfer matrices
function tN(SS_left::AbstractTensorMap{E, S, 2, 2}, SS_right::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    return transpose(SS_right * SS_left, ((3, 1), (4, 2)))
end

# Function to compute the vector W for a given position in the loop
function tW(
        pos::Int, psiA::Vector{TA}, psiB::Vector{TB},
        TSS_left::AbstractTensorMap{E, S, 2, 2}, TSS_right::AbstractTensorMap{E, S, 2, 2}
    ) where {
        TA <: AbstractTensorMap{<:Any, <:Any, 1, 3},
        TB <: AbstractTensorMap{<:Any, <:Any, 1, 2}, E, S,
    }
    ΨA = psiA[(pos - 1) ÷ 2 + 1]

    tmp = TSS_right * TSS_left

    if iseven(pos)
        ΨB = psiB[pos - 1]
        #--2---ΨB--1'-   --2'---------2--
        #      |       |         |
        #      4       3'        t
        #       |     |          m
        #        |   |           p
        #         | |            |
        #---3------ΨA------1----------3--
        @plansor W[-1; -3 -2] := conj(ΨB[2; 4 -1]) * ΨA[3; 4 -3 1] * tmp[-2 1; 2 3]
    else
        ΨB = psiB[pos + 1]
        #-1'--   --2'--ΨB--2----------1'-
        #      |       |         |
        #      3'      4         t
        #       |     |          m
        #        |   |           p
        #         | |            |
        #---3------ΨA------1----------3--
        @plansor W[-1; -3 -2] := conj(ΨB[-2; 4 2]) * ΨA[3; -3 4 1] * tmp[2 1; -1 3]
    end

    return transpose(W, ((1, 3), (2,)))
end

function opt_T(
        N::AbstractTensorMap{E, S, 2, 2}, W::AbstractTensorMap{E, S, 2, 1},
        psi::AbstractTensorMap{E, S, 2, 1}, loop_condition::LoopParameters
    ) where {E, S}
    if loop_condition.krylov == false
        ΔW = W - N * psi
        Δpsi = N \ ΔW
        new_psi = psi + Δpsi
        res = norm(N * Δpsi - ΔW)
        relative_shift = norm(Δpsi) / norm(psi)
        return new_psi, res, relative_shift
    elseif loop_condition.krylov == true
        new_psi, info = linsolve(
            x -> N * x, W, psi, loop_condition.krylovalg
        )
        if info.converged == 0
            @warn "The linsolve did not converge after $(info.numiter) iterations."
        end
        res = info.normres
        relative_shift = norm(new_psi - psi) / norm(psi)
        return new_psi, res, relative_shift
    end
end

# Function to compute the right cache for the transfer matrices. Here we sweep from left to right. At the end we add the identity transfer matrix to the cache.
# cache[1] = T2 * T3 * T4 * T5 * T6 * T7 * T8
# cache[2] = T3 * T4 * T5 * T6 * T7 * T8
# cache[3] = T4 * T5 * T6 * T7 * T8
# ...
# cache[7] = T8
# cache[8] = I
function right_cache(transfer_mats::Vector{T}) where {T <: AbstractTensorMap{E, S, 2, 2}} where {E, S}
    n = length(transfer_mats)
    cache = similar(transfer_mats)
    cache[end] = id(E, domain(transfer_mats[end]))

    for i in (n - 1):-1:1
        cache[i] = transfer_mats[i + 1] * cache[i + 1]
    end

    return cache
end

# A general function to optimize the truncation error of an MPS on a ring.
# Sweeping from left to right, we optimize the tensors in the loop by minimizing the cost function.
# Here cache of right-half-chain is used to minimize the number of multiplications to accelerate the sweeping.
# The transfer matrix on the left is updated after each optimization step.
# The cache technique is from Chenfeng Bao's thesis, see http://hdl.handle.net/10012/14674.
function loop_opt(
        psiA::Vector{T},
        trunc::TruncationStrategy,
        loop_condition::LoopParameters,
        verbosity::Int
    ) where {T <: AbstractTensorMap{E, S, 1, 3}} where {E, S}
    psiB = Ψ_B(psiA, trunc, loop_condition)
    if loop_condition.nuclear_norm
        M = map(x -> zeros(E, space(x)), psiB)
        Λ = copy(M)
        ξ = loop_condition.ξ_init
    end

    NB = length(psiB) # Number of tensors in the MPS Ψ_B
    psiBpsiB = ΨBΨB(psiB)
    psiBpsiA = ΨBΨA(psiB, psiA)
    psiApsiA = ΨAΨA(psiA)
    C = tr(reduce(*, psiApsiA)) # Since C is not changed during the optimization, we can compute it once and use it in the cost function.
    cost = Float64[Inf]

    sweep = 0
    crit = true
    while crit
        right_cache_BB = right_cache(psiBpsiB)
        right_cache_BA = right_cache(psiBpsiA)
        left_BB = id(E, codomain(psiBpsiB[1])) # Initialize the left transfer matrix for ΨBΨB
        left_BA = id(E, codomain(psiBpsiA[1])) # Initialize the left transfer matrix for ΨBΨA

        t_start = time()

        if sweep == 0
            tNt = tr(psiBpsiB[1] * right_cache_BB[1])
            tdw = tr(psiBpsiA[1] * right_cache_BA[1])
            wdt = conj(tdw)
            cost_this = real((C + tNt - wdt - tdw) / C)

            @infov 3 "Initial cost: $cost_this"

            push!(cost, cost_this)
        end

        crit = loop_condition.sweeping(sweep, cost)

        !crit && break

        for pos_psiB in 1:NB
            pos_psiA = (pos_psiB - 1) ÷ 2 + 1 # Position in the MPS Ψ_A

            N = tN(left_BB, right_cache_BB[pos_psiB]) # Compute the half of the matrix N for the current position in the loop, right cache is used to minimize the number of multiplications
            W = tW(pos_psiB, psiA, psiB, left_BA, right_cache_BA[pos_psiA]) # Compute the vector W for the current position in the loop, using the right cache for ΨBΨA
            psi = transpose(psiB[pos_psiB], ((1, 3), (2,)))

            if loop_condition.nuclear_norm
                N_eff = N + ξ * id(domain(N))
                W_eff = W + ξ * transpose(M[pos_psiB], ((1, 3), (2,))) + transpose(Λ[pos_psiB], ((1, 3), (2,)))
            else
                N_eff = N
                W_eff = W
            end

            new_psi, residual, relative_shift = opt_T(N_eff, W_eff, psi, loop_condition) # Optimize the tensor T for the current position in the loop, with the psiB[pos_psiB] be the initial guess
            psiB[pos_psiB] = transpose(new_psi, ((1,), (3, 2)))

            if loop_condition.nuclear_norm
                if iseven(pos_psiB)
                    M[pos_psiB], rank, nuclear_norm1 = singular_value_thresholding(psiB[pos_psiB] + (-Λ[pos_psiB] / ξ), ξ)
                else
                    new_M_transp, rank, nuclear_norm1 = singular_value_thresholding(transpose(psiB[pos_psiB], ((2, 1), (3,))) + (-transpose(Λ[pos_psiB], ((2, 1), (3,))) / ξ), ξ)
                    M[pos_psiB] = transpose(new_M_transp, ((2,), (1, 3)))
                end
                Λ[pos_psiB] += ξ * (M[pos_psiB] - psiB[pos_psiB])
            end

            @infov 4 "      ΔΨB[$pos_psiB] = $relative_shift, residual = $residual"

            @plansor BB_temp[-1 -2; -3 -4] := psiB[pos_psiB][-2; 1 -4] * conj(psiB[pos_psiB][-1; 1 -3])
            psiBpsiB[pos_psiB] = BB_temp # Update the transfer matrix for ΨBΨB
            left_BB = left_BB * BB_temp # Update the left transfer matrix for ΨBΨB

            if iseven(pos_psiB) # If the position is even, we also update the transfer matrix for ΨBΨA
                @plansor BA_temp[-1 -2; -3 -4] :=
                    conj(psiB[2 * pos_psiA - 1][-1; 1 3]) *
                    psiA[pos_psiA][-2; 1 2 -4] *
                    conj(psiB[2 * pos_psiA][3; 2 -3])
                psiBpsiA[pos_psiA] = BA_temp # Update the transfer matrix for ΨBΨA
                left_BA = left_BA * BA_temp # Update the left transfer matrix for ΨBΨA
            end
        end
        sweep += 1

        tNt = tr(left_BB)
        tdw = tr(left_BA)
        wdt = conj(tdw)
        cost_this = real((C + tNt - wdt - tdw) / C)
        push!(cost, cost_this)
        crit = loop_condition.sweeping(sweep, cost)

        @infov 3 "Sweep: $sweep, Cost: $(cost[end]), Time: $(time() - t_start)s" # Included the time taken for the sweep

        if loop_condition.nuclear_norm
            ξ = max(loop_condition.ρ * ξ, loop_condition.ξ_min)
        end
    end

    return psiB
end

"""
Coarse-grain `ΨB` to renormalized `TA`, `TB` tensors
"""
function ΨB_to_TATB(psiB::Vector{T}) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 2}}
    #= 
    (4)         (2)     (4)         (2)
      ↘        ↗          ↘        ↗
        7 --- 6             4 --- 1
        |  A  |             |  B  |
        2 --- 3             5 --- 8
      ↗        ↘          ↗        ↘
    (3)         (1)     (3)         (1)
    =#
    @plansor TA[-1 -2; -3 -4] := psiB[6][-2; 1 2] * psiB[7][2; 3 -4] *
        psiB[2][-3; 3 4] * psiB[3][4; 1 -1]
    @plansor TB[-1 -2; -3 -4] := psiB[1][1; 2 -2] * psiB[4][-4; 2 3] *
        psiB[5][3; 4 -3] * psiB[8][-1; 4 1]
    @assert _check_dual(TA) && _check_dual(TB)
    return TA, TB
end

function loop_opt!(
        scheme::LoopTNR, trunc::TruncationStrategy,
        loop_condition::LoopParameters,
        verbosity::Int
    )
    psiA = Ψ_A(scheme)
    psiB = loop_opt(psiA, trunc, loop_condition, verbosity)
    scheme.TA, scheme.TB = ΨB_to_TATB(psiB)
    return scheme
end

function step!(
        scheme::LoopTNR,
        trunc::TruncationStrategy,
        entanglement_criterion::stopcrit,
        loop_condition::LoopParameters,
        verbosity::Int
    )
    if !loop_condition.nuclear_norm
        entanglement_filtering!(scheme, loop_condition.truncentanglement, entanglement_criterion)
    end
    scheme = loop_opt!(scheme, trunc, loop_condition, verbosity)
    return scheme
end

function step!(
        scheme::LoopTNR,
        trunc::TruncationStrategy,
        loop_condition::LoopParameters,
        verbosity::Int
    )
    return step!(scheme, trunc, default_entanglement_criterion, loop_condition, verbosity)
end

function run!(
        scheme::LoopTNR, trscheme::TruncationStrategy,
        criterion::stopcrit, loop_condition::LoopParameters,
        finalizer::Finalizer{E};
        entanglement_criterion = default_entanglement_criterion,
        finalize_beginning = true,
        verbosity = 1
    ) where {E}
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
            step!(scheme, trscheme, entanglement_criterion, loop_condition, verbosity)
            push!(data, finalizer.f!(scheme))

            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

function run!(scheme, trscheme, criterion, loop_condition; kwargs...)
    return run!(scheme, trscheme, criterion, loop_condition, default_Finalizer; kwargs...)
end

function run!(
        scheme::LoopTNR, trscheme::TruncationStrategy, criterion::stopcrit;
        finalize_beginning = true, verbosity = 1
    )
    loop_condition = LoopParameters()
    return run!(
        scheme, trscheme, criterion, loop_condition;
        finalize_beginning = finalize_beginning,
        verbosity = verbosity
    )
end

function Base.show(io::IO, scheme::LoopTNR)
    println(io, "LoopTNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end
