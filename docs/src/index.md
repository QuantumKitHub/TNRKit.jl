# TNRKit

**Your one-stop-shop for Tensor Network Renormalization.**

# Package summary
TNRKit.jl aims to provide as many Tensor Network Renormalization methods as possible. Several models like the classical Ising, Potts and Six Vertex models are provided.

You can use TNRKit for calculating:
1. Partition functions (classical & quantum)
2. CFT data
3. Central charges

Many common TNR schemes have already been implemented:

**2D square tensor networks**
* [`TRG`](@ref) (Levin and Nave's Tensor Renormalization Group)
* [`BTRG`](@ref) (bond-weighted TRG)
* [`LoopTNR`](@ref) (entanglement filtering + loop optimization)
* [`SLoopTNR`](@ref) (c4 & inversion symmetric LoopTNR)
* [`HOTRG`](@ref) (higher order TRG)
* [`ATRG`](@ref) (anisotropic TRG)

**2D square CTM methods**
* [`CTM`](@ref) (Corner Transfer Matrix)
* [`c4vCTM`](@ref) (c4v symmetric CTM)
* [`rCTM`](@ref) (reflection symmetric CTM)
* [`ctm_TRG`](@ref) (Corner Transfer Matrix environment + TRG)
* [`ctm_HOTRG`](@ref) (Corner Transfer Matrix environment + HOTRG)

**2D triangular CTM methods**
* [`c6vCTM_triangular`](@ref) (c6v symmetric CTM on the triangular lattice)
* [`CTM_triangular`](@ref) (CTM on the triangular lattice)

**2D honeycomb CTM methods**
* [`c3vCTM_honeycomb`](@ref) (c3v symmetric CTM on the honeycomb lattice)
* [`CTM_honeycomb`](@ref) (CTM on the honeycomb lattice)

**Impurity Methods**
* [`ImpurityTRG`](@ref)
* [`ImpurityHOTRG`](@ref)

**3D cubic tensor networks**
* [`ATRG_3D`](@ref) (anisotropic TRG)
* [`HOTRG_3D`](@ref) (higher order TRG)

# Quick Start Guide
1. Choose a (TensorKit!) tensor that respects the leg-convention (see below)
2. Choose a TNR scheme
3. Choose a truncation scheme
4. Choose a stopping criterion

For example:
```julia
using TNRKit, TensorKit

T = classical_ising(ising_βc) # partition function of classical Ising model at the critical point
scheme = BTRG(T) # Bond-weighted TRG (excellent choice)
data = run!(scheme, truncrank(16), maxiter(25)) # max bond-dimension of 16, for 25 iterations
```
`data` now contains 26 norms of the tensor, 1 for every time the tensor was normalized. (By default there is a normalization step before the first coarse-graining step wich can be turned off by changing the kwarg `run!(...; finalize_beginning=false)`)

Using these norms you could, for example, calculate the free energy of the critical classical Ising model:
```Julia
f = free_energy(data, ising_βc) # -2.1096504926141826902647832
```
You could even compare to the exact value, as calculated by the [Onsager solution](https://en.wikipedia.org/wiki/Ising_model#:~:text=Onsager%27s%20exact%20solution):

```julia-repl
julia> abs((f - f_onsager) / f_onsager)
3.1e-07
```
Pretty impressive for a calculation that takes about 0.3s on a laptop.

## Verbosity
There are 3 levels of verbosity implemented in TNRKit:
- Level 0: no TNRKit messages whatsoever.
- Level 1: Info at beginning and end of the simulations (including information on why the simulation stopped, how long it took and how many iterations were performed).
- Level 2: Level 1 + info at every iteration about the last generated finalize output and the iteration number.
  
to choose the verbosity level, simply use `run!(...; verbosity=n)`. The default is `verbosity=1`.

## Included Models on the square lattice
TNRKit includes several common models out of the box.
- Ising model in 2D: `classical_ising(S, β; h=0)` where `S` can be `Trivial` or `Z2Irrep` to specify the symmetry.
- Ising model in 2D with impurities: `classical_ising_impurity(β; h=0)`.
- Ising model in 3D: `classical_ising_3D(S, β; h=0)` where `S` can be `Trivial` or `Z2Irrep` to specify the symmetry.
- Potts model in 2D: `classical_potts(S, q, β)`, where `S` can be `Trivial` or `ZNIrrep{q}` to specify the symmetry.
- Potts model in 2D with impurities: `classical_potts_impurity(q, β)`.
- Six Vertex model: `sixvertex(S, elt; a=1.0, b=1.0, c=1.0)` where `S` can be `Trivial`, `U1Irrep` or `CU1Irrep` to specify the symmetry and `elt` can be any number type (default is `Float64`).
- Clock model: `classical_clock(S, q, β)` where `S` can be `Trivial`, `ZNIrrep{q}` or `DNIrrep{q}` to specify the symmetry.
- XY model in 2D: `classical_XY(S, β, charge_trunc)` where `S` can be `U1Irrep` or `CU1Irrep` to specify the symmetry.
- Real $\phi^4$ model: `phi4_real(S, K, μ0, λ, h)` where `S` can be `Trivial` or `Z2Irrep` to specify the symmetry.
- Real $\phi^4$ model with impurities: `phi4_real_imp1(S, K, μ0, λ, h)` and `phi4_real_imp2(S, K, μ0, λ, h)` where `S` can be `Trivial`.
- Complex $\phi^4$ model: `phi4_complex(S, K, μ0, λ)` where `S` can be `Trivial`, `Z2Irrep ⊠ Z2Irrep` or `U1Irrep` to specify the symmetry.
- Gross-Neveu model: `gross_neveu_start(S, μ, m, g)` where `S` can be `FermionParity` to specify the symmetry.

## Included Models on the triangular lattice
TNRKit includes several common models out of the box.
- Ising model: `classical_ising_triangular(S, β; h=0)` where `S` can be `Trivial` or `Z2Irrep` to specify the symmetry.

## Included Models on the honeycomb lattice
TNRKit includes several common models out of the box.
- Ising model: `classical_ising_honeycomb(S, β; h=0)` where `S` can be `Trivial` or `Z2Irrep` to specify the symmetry.
  
If you want to implement your own model you must respect the leg-convention assumed by all TNRKit schemes.

## Leg-convention
All the schemes assume that the input tensor lives in the space $V_1 \otimes V_2 \leftarrow V_3 \otimes V_4$ and that the legs are ordered in the following way:
```
     3
     |
     v
     |
1-<--┼--<-4
     |
     v
     |
     2
```

The 3D scheme(s) assume that the input tensor lives in the space $V_{\text{D}} \otimes V_{\text{U}} \prime \leftarrow V_{\text{N}} \otimes V_{\text{E}} \otimes V_{\text{S}} \prime \otimes V_{\text{W}} \prime$.

Where D, U, N, E, S, W stand for Down, Up, North, East, South and West.