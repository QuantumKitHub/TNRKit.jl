![TNRKit Logo](https://github.com/VictorVanthilt/TNRKit.jl/blob/master/docs/src/assets/logo-dark.svg#gh-dark-mode-only)
![TNRKit Logo](https://github.com/VictorVanthilt/TNRKit.jl/blob/master/docs/src/assets/logo.svg#gh-light-mode-only)

# TNRKit.jl
| **Documentation** | **Build Status** | **Digital Object Identifyer** | **Coverage** |
|:-----------------:|:----------------:|:-----------------------------:|:------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url] | [![DOI][doi-img]][doi-url] | [![Codecov][codecov-img]][codecov-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://VictorVanthilt.github.io/TNRKit.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://VictorVanthilt.github.io/TNRKit.jl/dev

[ci-img]: https://github.com/VictorVanthilt/TNRKit.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/VictorVanthilt/TNRKit.jl/actions/workflows/CI.yml

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.16836269.svg
[doi-url]: https://doi.org/10.5281/zenodo.16836269

[codecov-img]: https://codecov.io/gh/VictorVanthilt/TNRKit.jl/graph/badge.svg?token=XEOJODNBF3
[codecov-url]: https://codecov.io/gh/VictorVanthilt/TNRKit.jl


TNRKit.jl is a Julia package that aims to implement as many tensor network renormalization (TNR) schemes as possible.
It is built upon
[TensorKit.jl](https://github.com/jutho/TensorKit.jl), which provides functionality for symmetric tensors.
The following schemes are currently implemented:

**2D square tensor networks**
- TRG (Levin and Nave's Tensor Renormalization Group)
- BTRG (bond-weighted TRG)
- LoopTNR (entanglement filtering + loop optimization)
- SLoopTNR (C4 & inversion symmetric LoopTNR)
- HOTRG (higher order TRG)
- ATRG (anisotropic TRG)

**2D square CTM methods**
- CTM (Corner Transfer Matrix)
- c4vCTM (c4v symmetric CTM)
- rCTM (reflection symmetric CTM)
- ctm_TRG (Corner Transfer Matrix environment + TRG)
- ctm_HOTRG (Corner Transfer Matrix environment + HOTRG)

**2D triangular CTM methods**
- c6vCTM_triangular (c6v symmetric CTM on the triangular lattice)
- CTM_triangular (CTM on the triangular lattice)

**2D honeycomb CTM methods**
- c3vCTM_honeycomb (c3v symmetric CTM on the honeycomb lattice)
- CTM_honeycomb (CTM on the honeycomb lattice)

**2D Impurity Methods**
- ImpurityTRG (Expectation value calculation via TRG)
- ImpurityHOTRG (Expectation value calculation via HOTRG)

**2D Correlation Methods**
- CorrelationHOTRG (Correlation function calculation via HOTRG)

**3D cubic tensor networks**
- ATRG_3D (anisotropic TRG)
- HOTRG_3D (higher order TRG)

This project is under active development. The interface is subject to changes. Any feedback about the user interface or the internals is much appreciated. The github discussions page is a great place to talk!

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

To choose the verbosity level, simply use `run!(...; verbosity=n)`. The default is `verbosity=1`.

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
