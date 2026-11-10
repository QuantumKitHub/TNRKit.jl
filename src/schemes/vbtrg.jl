"""
$(TYPEDEF)

Variational Boundary Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T)

# Running the algorithm
    run!(::VBTRG, trunc::TruncationByOrder, stop::Stopcrit[, finalizer=default_Finalizer];
         inner_trunc=truncrank(2 * trunc.howmany), envdim=trunc.howmany, boundary_alg=VUMPS(; verbosity=0),
         vumps_steps=1, projector_maxiter=50, projector_tol=1e-10, finalize_beginning=true, verbosity=1)

Each step coarse-grains a 2×2 block of tensors into a single tensor (the lattice is rescaled by a linear factor of 2),
so `free_energy` should be called with `scalefactor = 4.0`.
The bond-merging projectors are not obtained from local SVDs, as in HOTRG, but are optimized with respect to the
global environment, approximated by the fixed-point boundary MPS (and its channel environments) of the row-to-row
transfer matrix. The coarse-graining of the 2×2 block is made cheap by first splitting every tensor along a diagonal
and truncating the resulting bond with globally optimized intermediate projectors.

The boundary MPS and its environments are updated with the vertical and horizontal projectors after each step, and
then refined with `vumps_steps` iterations of VUMPS on the coarse-grained tensor (the reference uses 0 or 1).

* `trunc`: bond dimension `χ` of the coarse-grained tensor (`χh = χv = χ` in the reference).
* `inner_trunc`: bond dimension `χq` of the intermediate projectors.
* `envdim`: bond dimension `D` of the boundary MPS.

Only real tensors with `Trivial` symmetry are supported, and the transfer matrix is assumed to be symmetric under
vertical reflection, so that the lower boundary is the conjugate of the upper boundary.

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step
    - 3: Print convergence information of the projector optimization

# Fields

$(TYPEDFIELDS)

# References
* [Song & Kawashima arXiv:2508.10418 (2025)](@cite song2025)
"""
mutable struct VBTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "central tensor"
    T::TT

    function VBTRG(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        @assert sectortype(T) == Trivial "Only trivial sector type is supported for the tensor T"
        return new{E, S, TT}(T)
    end
end

function Base.show(io::IO, scheme::VBTRG)
    println(io, "VBTRG - Variational Boundary Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end

# Conventions:
# - The boundary MPS `mps` is the fixed point of the row-to-row transfer matrix `InfiniteMPO([T])`: its physical legs
#   connect to the North legs of `T`, the lower boundary is `conj(mps)`, connecting to the South legs.
# - `GL[a m; b]` and `GR[b m; a]` (MPSKit `InfiniteEnvironments`): `b` connects to the upper boundary, `a` to the
#   lower boundary and `m` to the West (`GL`) or East (`GR`) leg of `T`.
# - Every projector is of the form `w†w` or `ww†`, with `w` an isometry that maximizes `tr(Ω w w†)`, where `Ω` is
#   the environment of the projector. `Γ(x)` is `Ω` contracted with `w†`, such that `tr(Ω w w†) = tr(w * Γ(x))`.

# Vertical projector code: P_v = w† w merges the North legs of two horizontally adjacent tensors.
mutable struct _vbtrg_vertical_projector{E, S}
    mps::MPSKit.InfiniteMPS
    w::AbstractTensorMap{E, S, 1, 2}
    function _vbtrg_vertical_projector(mps::MPSKit.InfiniteMPS, χv::Int)
        w = randn(scalartype(mps), spacetype(mps)(χv) ← space(mps.AL[1], 2) ⊗ space(mps.AC[1], 2))
        return new{scalartype(w), spacetype(w)}(mps, w)
    end
end

function trΩ(x::_vbtrg_vertical_projector)
    return norm(x.mps.AC[1])^2
end

function trΩ_w_dagw(x::_vbtrg_vertical_projector)
    return @tensoropt x.mps.AL[1][7 1; 2] * x.mps.AC[1][2 3; 9] * x.w[8; 1 3] * conj(x.w[8; 4 5]) * conj(x.mps.AL[1][7 4; 6]) * conj(x.mps.AC[1][6 5; 9])
end

function Γ(x::_vbtrg_vertical_projector)
    return @tensoropt Γ[-2 -3; -1] := x.mps.AL[1][4 -2; 6] * x.mps.AC[1][6 -3; 5] * conj(x.w[-1; 2 3]) * conj(x.mps.AL[1][4 2; 1]) * conj(x.mps.AC[1][1 3; 5])
end

# Horizontal projector code: P_h = w w† merges the East legs of two vertically adjacent tensors.
# The environment of the two rows is approximated by two stacked channel environments (top and bottom),
# glued along their middle virtual bond, with the bond matrix C closing the top and bottom boundary.
mutable struct _vbtrg_horizontal_projector{E, S}
    mps::MPSKit.InfiniteMPS
    env::MPSKit.InfiniteEnvironments
    w::AbstractTensorMap{E, S, 2, 1}
    function _vbtrg_horizontal_projector(mps::MPSKit.InfiniteMPS, env::MPSKit.InfiniteEnvironments, χh::Int)
        Vh = dual(space(env.GLs[1], 2))
        w = randn(scalartype(mps), Vh ⊗ Vh ← spacetype(mps)(χh))
        return new{scalartype(w), spacetype(w)}(mps, env, w)
    end
end

function trΩ(x::_vbtrg_horizontal_projector)
    GL, GR, C = x.env.GLs[1], x.env.GRs[1], x.mps.C[1]
    return @tensoropt GL[1 2; 3] * GL[4 5; 1] * C[3; 6] * GR[6 2; 7] * GR[7 5; 8] * conj(C[4; 8])
end

function trΩ_w_dagw(x::_vbtrg_horizontal_projector)
    GL, GR, C = x.env.GLs[1], x.env.GRs[1], x.mps.C[1]
    return @tensoropt GL[1 2; 3] * GL[4 5; 1] * C[3; 6] * GR[6 9; 7] * GR[7 10; 8] * conj(C[4; 8]) * x.w[2 5; 11] * conj(x.w[9 10; 11])
end

function Γ(x::_vbtrg_horizontal_projector)
    GL, GR, C = x.env.GLs[1], x.env.GRs[1], x.mps.C[1]
    return @tensoropt Γ[-3; -1 -2] := GL[1 -1; 3] * GL[4 -2; 1] * C[3; 6] * GR[6 9; 7] * GR[7 10; 8] * conj(C[4; 8]) * conj(x.w[9 10; -3])
end

# Intermediate projector code
# T is split along a diagonal without truncation, T = Tbl * Ttr, and the intermediate bond is truncated by the
# projector w† w, optimized in the single-site environment of T.
mutable struct _vbtrg_intermediate_projector_trbl{E, S}
    mps::MPSKit.InfiniteMPS
    env::MPSKit.InfiniteEnvironments
    Ttr::AbstractTensorMap{E, S, 1, 2}
    Tbl::AbstractTensorMap{E, S, 2, 1}
    w::AbstractTensorMap{E, S, 1, 1}
    function _vbtrg_intermediate_projector_trbl(mps::MPSKit.InfiniteMPS, env::MPSKit.InfiniteEnvironments, T::AbstractTensorMap{E, S, 2, 2}, χq::Int) where {E, S}
        Tbl, Ttr = SVD12(T, notrunc())
        w = randn(scalartype(mps), spacetype(mps)(χq) ← codomain(Ttr)[1])
        return new{scalartype(w), spacetype(w)}(mps, env, Ttr, Tbl, w)
    end
end

function trΩ(x::_vbtrg_intermediate_projector_trbl)
    return @tensoropt x.env.GLs[1][1 3; 7] * x.mps.AC[1][7 5; 4] * x.env.GRs[1][4 6; 9] * x.Ttr[8; 5 6] * x.Tbl[3 2; 8] * conj(x.mps.AC[1][1 2; 9])
end

function trΩ_w_dagw(x::_vbtrg_intermediate_projector_trbl)
    return @tensoropt x.env.GLs[1][2 1; 9] * x.mps.AC[1][9 5; 7] * x.env.GRs[1][7 6; 11] * x.Ttr[8; 5 6] * x.w[10; 8] * conj(x.w[10; 4]) * x.Tbl[1 3; 4] * conj(x.mps.AC[1][2 3; 11])
end

function Γ(x::_vbtrg_intermediate_projector_trbl)
    return @tensoropt Γ[-1; -2] := x.env.GLs[1][2 1; 8] * x.mps.AC[1][8 7; 5] * x.Ttr[-1; 7 6] * x.env.GRs[1][5 6; 9] * conj(x.mps.AC[1][2 3; 9]) * x.Tbl[1 3; 4] * conj(x.w[-2; 4])
end

# T = Ttl * Tbr, with the intermediate bond truncated by the projector w w†.
mutable struct _vbtrg_intermediate_projector_tlbr{E, S}
    mps::MPSKit.InfiniteMPS
    env::MPSKit.InfiniteEnvironments
    Ttl::AbstractTensorMap{E, S, 1, 2}
    Tbr::AbstractTensorMap{E, S, 2, 1}
    w::AbstractTensorMap{E, S, 1, 1}
    function _vbtrg_intermediate_projector_tlbr(mps::MPSKit.InfiniteMPS, env::MPSKit.InfiniteEnvironments, T::AbstractTensorMap{E, S, 2, 2}, χq::Int) where {E, S}
        Ttl, Tbr = SVD12(permute(T, ((1, 3), (2, 4))), notrunc())
        Ttl = permute(Ttl, ((1,), (2, 3)))
        Tbr = permute(Tbr, ((1, 2), (3,)))
        w = randn(scalartype(mps), domain(Ttl)[2] ← spacetype(mps)(χq))
        return new{scalartype(w), spacetype(w)}(mps, env, Ttl, Tbr, w)
    end
end

function trΩ(x::_vbtrg_intermediate_projector_tlbr)
    return @tensoropt x.env.GLs[1][1 2; 3] * x.mps.AC[1][3 4; 5] * x.env.GRs[1][5 8; 6] * x.Ttl[2; 4 9] * x.Tbr[9 7; 8] * conj(x.mps.AC[1][1 7; 6])
end

function trΩ_w_dagw(x::_vbtrg_intermediate_projector_tlbr)
    return @tensoropt x.env.GLs[1][1 2; 3] * x.mps.AC[1][3 4; 5] * x.env.GRs[1][5 8; 6] * x.Ttl[2; 4 11] * x.w[11; 10] * conj(x.w[9; 10]) * x.Tbr[9 7; 8] * conj(x.mps.AC[1][1 7; 6])
end

function Γ(x::_vbtrg_intermediate_projector_tlbr)
    return @tensoropt Γ[-1; -2] := x.env.GLs[1][1 2; 3] * x.mps.AC[1][3 4; 5] * x.Ttl[2; 4 -2] * x.env.GRs[1][5 6; 7] * conj(x.mps.AC[1][1 8; 7]) * x.Tbr[9 8; 6] * conj(x.w[9; -1])
end

function run!(
        scheme::VBTRG, trunc::MatrixAlgebraKit.TruncationByOrder, criterion::stopcrit, finalizer::Finalizer{E};
        inner_trunc::MatrixAlgebraKit.TruncationByOrder = truncrank(2 * trunc.howmany), envdim::Int = trunc.howmany,
        boundary_alg = VUMPS(; verbosity = 0), vumps_steps::Int = 1, projector_maxiter::Int = 50,
        projector_tol::Real = 1.0e-10, finalize_beginning = true, verbosity = 1
    ) where {E}
    data = Vector{E}()
    projector_kwargs = (; maxiter = projector_maxiter, convcrit = projector_tol)

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, finalizer.f!(scheme))
        end

        # 1. Initialize the boundary MPS and its environments with VUMPS
        mps = InfiniteMPS(randn, scalartype(scheme.T), domain(scheme.T)[1], spacetype(scheme.T)(envdim))
        mps, env, _ = leading_boundary(mps, InfiniteMPO([scheme.T]), boundary_alg)

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            # 2. Optimize the projectors in the boundary environment
            vert = _vbtrg_vertical_projector(mps, trunc.howmany)
            optimize!(vert, trunc; projector_kwargs...)

            horiz = _vbtrg_horizontal_projector(mps, env, trunc.howmany)
            optimize!(horiz, trunc; projector_kwargs...)

            qstruct1 = _vbtrg_intermediate_projector_trbl(mps, env, scheme.T, inner_trunc.howmany)
            optimize!(qstruct1, inner_trunc; projector_kwargs...)

            qstruct2 = _vbtrg_intermediate_projector_tlbr(mps, env, scheme.T, inner_trunc.howmany)
            optimize!(qstruct2, inner_trunc; projector_kwargs...)

            # 3. Coarse-grain the tensor and project the boundary MPS and its environments
            _update_tensor!(scheme, vert, horiz, qstruct1, qstruct2)
            mps, env = _update_boundary(mps, env, vert, horiz)

            # 4. Refine the boundary with VUMPS on the coarse-grained tensor
            if vumps_steps > 0
                mps, env, _ = leading_boundary(mps, InfiniteMPO([scheme.T]), VUMPS(; maxiter = vumps_steps, verbosity = 0), env)
            end

            push!(data, finalizer.f!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

run!(scheme::VBTRG, trunc::MatrixAlgebraKit.TruncationByOrder, criterion::stopcrit; kwargs...) = run!(scheme, trunc, criterion, default_Finalizer; kwargs...)

const VBTRG_PROJECTOR = Union{_vbtrg_vertical_projector, _vbtrg_horizontal_projector, _vbtrg_intermediate_projector_trbl, _vbtrg_intermediate_projector_tlbr}

function _update_projector!(proj::VBTRG_PROJECTOR, trunc::TruncationStrategy)
    U, S, Vdag, _ = svd_trunc(Γ(proj); trunc = trunc)
    proj.w = adjoint(Vdag) * adjoint(U)

    # rescale such that tr(Ω w w†) = tr(Ω)
    C1 = trΩ(proj)
    C2 = trΩ_w_dagw(proj)
    proj.w = proj.w * sqrt(C1 / C2)
    return S
end

function optimize!(proj::VBTRG_PROJECTOR, trunc::TruncationStrategy; maxiter::Int = 50, convcrit::Real = 1.0e-10)
    S = _update_projector!(proj, trunc)

    for iter in 2:maxiter
        Scurr = _update_projector!(proj, trunc)

        if space(S) == space(Scurr)
            normdiff = norm(S - Scurr) / norm(Scurr)
            if normdiff <= convcrit
                @infov 3 "optimize! converged in $iter iterations | normdiff = $normdiff"
                return proj
            end
        end
        S = Scurr
    end
    @warn "optimize! did not converge in $maxiter iterations"
    return proj
end

# Merge two sites of the boundary MPS and two rows of the channel environments with the projectors:
# A_L′ = A_L A_L w_v, A_R′ = A_R A_R w_v, G_L′ = (G_L G_L) w_h and G_R′ = (G_R G_R) w_h†.
# A_L′ and A_R′ are related by the old bond matrix, A_L′ C = C A_R′, but are no longer isometric.
# They are brought back into canonical form, and the environments are transformed along with the gauge.
function _update_boundary(mps::MPSKit.InfiniteMPS, env::MPSKit.InfiniteEnvironments, vert::_vbtrg_vertical_projector, horiz::_vbtrg_horizontal_projector)
    AL, C = mps.AL[1], mps.C[1]
    GL, GR = env.GLs[1], env.GRs[1]
    @tensoropt AL′[-1 -2; -3] := AL[-1 1; 2] * AL[2 3; -3] * vert.w[-2; 1 3]
    @tensoropt GL′[-1 -2; -3] := GL[1 2; -3] * GL[-1 3; 1] * horiz.w[2 3; -2]
    @tensoropt GR′[-1 -2; -3] := GR[-1 1; 2] * GR[2 3; -3] * conj(horiz.w[1 3; -2])

    # A_L″ = X A_L′ X⁻¹
    AL″, X, Xinv = _left_orthonormalize(AL′)
    mps″ = InfiniteMPS([AL″], C)

    # A_R″ = C″⁻¹ A_L″ C″ = Y A_R′ Y⁻¹
    Y = inv(mps″.C[1]) * X * C
    Yinv = inv(Y)

    @tensoropt GL″[-1 -2; -3] := GL′[1 -2; 2] * Xinv[2; -3] * conj(Xinv[1; -1])
    @tensoropt GR″[-1 -2; -3] := Y[-1; 1] * GR′[1 -2; 2] * conj(Y[-3; 2])

    # normalize the environments as in MPSKit, the projected environments grow exponentially otherwise
    C″ = mps″.C[1]
    GL″ /= norm(GL″)
    GR″ /= norm(GR″)
    n = @tensor GL″[1 2; 3] * C″[3; 4] * GR″[4 2; 5] * conj(C″[1; 5])
    GL″ /= n
    return mps″, MPSKit.InfiniteEnvironments(MPSKit.PeriodicVector([GL″]), MPSKit.PeriodicVector([GR″]))
end

# Find the gauge transformation X such that A_L = X A X⁻¹ is left isometric, from the left fixed point ρ = X†X.
function _left_orthonormalize(A::AbstractTensorMap{E, S, 2, 1}) where {E, S}
    ρ₀ = id(E, space(A, 1))
    vals, vecs, _ = eigsolve(ρ₀, 1, :LM) do ρ
        return @tensor ρ′[-1; -2] := ρ[1; 2] * A[2 3; -2] * conj(A[1 3; -1])
    end
    # fix the phase, the eigensolver can return a complex eigenvector for a real map
    ρ = vecs[1] / tr(vecs[1])
    ρ = E <: Real ? real(ρ) : ρ
    ρ = (ρ + ρ') / 2
    D, U = eigh_full(ρ)
    # ρ can become numerically rank deficient, regularize the smallest eigenvalues
    D.data .= max.(D.data, 1.0e-12 * maximum(D.data))
    sqrtD = sqrt(D)
    X = U * sqrtD * U'
    Xinv = U * inv(sqrtD) * U'
    @tensor AL[-1 -2; -3] := X[-1; 1] * A[1 -2; 2] * Xinv[2; -3]
    return AL / sqrt(abs(vals[1])), X, Xinv
end

function _update_tensor!(scheme::VBTRG, vert::_vbtrg_vertical_projector, horiz::_vbtrg_horizontal_projector, qstruct1::_vbtrg_intermediate_projector_trbl, qstruct2::_vbtrg_intermediate_projector_tlbr)
    # The 2×2 block has the trbl-split tensors on the top-left and bottom-right sites,
    # and the tlbr-split tensors on the top-right and bottom-left sites.

    # Make North block
    @tensoropt N[-1; -2 -3] := qstruct1.Ttr[4; 1 2] * qstruct2.Ttl[2; 3 5] * qstruct1.w[-1; 4] * qstruct2.w[5; -3] * conj(vert.w[-2; 1 3])

    # Make East block
    @tensoropt E[-1 -2; -3] := qstruct1.w[-1; 4] * qstruct1.Ttr[4; 2 3] * qstruct2.Tbr[5 2; 1] * horiz.w[1 3; -3] * conj(qstruct2.w[5; -2])

    # Make South block
    @tensoropt S[-1 -2; -3] := conj(qstruct2.w[1; -2]) * qstruct2.Tbr[1 2; 3] * qstruct1.Tbl[3 4; 5] * vert.w[-1; 2 4] * conj(qstruct1.w[-3; 5])

    # Make West block
    @tensoropt W[-1; -2 -3] := conj(qstruct1.w[-2; 1]) * qstruct1.Tbl[2 3; 1] * qstruct2.Ttl[4 3; 5] * qstruct2.w[5; -3] * conj(horiz.w[2 4; -1])

    # Combine blocks
    @tensoropt T[-1 -2; -3 -4] := N[1; -3 2] * E[3 2; -4] * S[-2 4; 3] * W[-1; 1 4]
    scheme.T = T
    return scheme
end
