# Conformal Field Theory Data

TNRKit provides extensive tools for calculating conformal field theory data. Details about the implementation can be found in the TNRKit paper ([arxiv/2604.06922](https://arxiv.org/abs/2604.06922)).

The core idea behind calculating the central charge, scaling dimensions, and conformal spins is to calculate spectra of **transfer matrices** constructed from fixed point tensors on a tube geometry. There are different ways to put fixed point tensors on a tube, and the geometry of this tube is characterised by 3 parameters:
$$[h, L, x]$$
where $h$ is the height of the tube, $L$ is the circumference, and $x$ is the horizontal shift. The higher the ratio $\frac{L}{h}$, the higher the resolution but also the more expensive the calculation.

To calculate CFT data we provide the `CFTData` struct, which can be used in the following ways:

```julia
CFTData(scheme; shape=[h, L, x])
CFTData(T::TensorMap; kwargs...) # 1 fixed point tensor
CFTData(TA::TensorMap, TB::TensorMap; kwargs...) # 2x2 checkerboard unitcell
```

The shapes we provide are: $[1, 1, 0]$, $[\sqrt{2}, 2\sqrt{2}, 0]$, $[1, 4, 1]$, $[1, 8, 1]$, $[\frac{4}{\sqrt{10}}, 2 \sqrt{10}, \frac{2}{\sqrt{10}}]$

The last two shapes require intermediate truncation steps, whose parameters can be tuned by:

```julia
CFTData(scheme; shape=[1, 8, 1], trunc = trunc1, truncentanglement=trunc2)
```

## CFTData Struct

The `CFTData` struct has three fields:

- `central_charge`: a complex number which is the central charge of the CFT.
- `modular_parameter`: a complex number which is the modular parameter of one tensor before building the tube transfer matrix.
- `scaling_dimensions`: a `StructuredVector` storing the scaling dimensions and conformal spins of the primary fields and their descendants. It can be indexed like an `AbstractVector` (i.e. with scalars, slices, ...), or with sectors (e.g. `Z2Irrep(0)`), which will provide the scaling dimensions associated with that sector/charge.
    To check which sectors you can index the `scaling_dimensions` with you can use `keys(scaling_dimensions)`.

## How Fermionic CFT Data Is Extracted

For a bosonic tensor network, `CFTData` builds a transfer matrix on the requested tube and diagonalizes it in each ordinary charge sector. For a fermionic tensor network, the same construction has one extra piece of global data: the **spin structure** around the tube. TNRKit extracts it by evaluating the same transfer matrix with two choices of boundary condition:

- `:R`: periodic fermions around the tube, obtained with `pbc = true`. The macro `@tensor` automatically inserts a fermionic twist to take the *supertrace* across the tube.
- `:NS`: antiperiodic fermions around the tube. Internally this is obtained by setting `pbc = false`, which explicitly inserts an additional fermionic twist to cancel the automatic supertrace twist, leaving the ordinary trace across the tube.

The result is a `StructuredVector` whose keys are tuples `(spin_structure, charge)`. For fermionic systems without additional symmetries besides the fermion parity, there will be four sectors:

```julia
(:NS, FermionParity(0))
(:NS, FermionParity(1))
(:R,  FermionParity(0))
(:R,  FermionParity(1))
```

The largest eigenvalue (corresponding to the identity field) should be in the NS even sector.
