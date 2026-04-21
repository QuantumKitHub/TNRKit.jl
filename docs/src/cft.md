# Conformal Field Theory Data
TNRKit provides extensive tools for calculating conformal field theory data. Details about the implementation can be found in the TNRKit paper ([arxiv/2604.06922](https://arxiv.org/abs/2604.06922)).

The core idea behind calculating the central charge, scaling dimensions, and conformal spins, is to calculate the spectrum of the fixed point tensor on a tube geometry. There are different ways to put fixed point tensors on a tube and the geometry of this tube is characterised by 3 parameters:
$$[h, L, x]$$
Where $h$ is the height of the tube, $L$ is the circumference, and $x$ is the horizontal shift. The higher the ratio $\frac{L}{h}$, the higher the resolution but also the more expensive the calculation.

To calculate cft data we provide the `CFTData` struct which can be used in the following ways:

```julia
CFTData(scheme; shape=[h, L, x])
CFTData(T::TensorMap; kwargs...) # 1 fixed point tensor
CFTData(TA::TensorMap, TB::TensorMap; kwargs...) # 2x2 checkerboard unitcell
```

The shapes we provide are: $[1, 1, 0]$, $[\sqrt{2}, 2\sqrt{2}, 0]$, $[1, 4, 1]$, $[1, 8, 1]$, $[\frac{4}{\sqrt{10}}, 2 \sqrt{10}, \frac{2}{\sqrt{10}}]$

The last two of which require intermediate truncation steps, the parameters of which can be tuned by:
```julia
CFTData(scheme; shape=[1, 8, 1], trunc = trunc1, truncentanglement=trunc2)
```

# CFTData struct
The `CFTData` struct has two fields:
- `central_charge`
- `scaling_dimensions`

The `central_charge` can be either `missing` (when using the $[1, 1, 0]$ shape), or a number.
The `scaling_dimensions` field is a `SectorVector` from TensorKit.jl.

The `scaling_dimensions` can be indexed like an `AbstractVector` (i.e. with scalars, slices, ...), or with sectors (e.g. `Z2Irrep(0)`), which will provide the scaling dimensions associated with that sector/charge.
To check which sectors you can index the `scaling_dimensions` with you can use `keys(scaling_dimensions)`.