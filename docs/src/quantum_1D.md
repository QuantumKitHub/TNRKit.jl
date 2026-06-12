# 1+1D Quantum Models

TNRKit provides tools to construct partition function tensors for $(1+1)$-dimensional quantum lattice models (currently only nearest-neighbor Hamiltonians are supported). These tensors can then be studied with any of the 2D tensor network renormalization schemes (TRG, HOTRG, LoopTNR, etc.).

## Quantum Partition Functions

A $(1+1)$D quantum system at inverse temperature $\beta$ has partition function

```math
\mathcal{Z} = \mathrm{Tr}\, e^{-\beta H}
```

where $H$ is the Hamiltonian. By discretizing the imaginary-time direction into $N$ steps of size $\delta\tau = \beta / N$ and applying a Trotter–Suzuki decomposition, the partition function becomes a contraction of a 2D tensor network. Each time step is represented by a Trotter gate acting on nearest-neighbor pairs.

## Tensor Network Representation of $\mathcal{Z}$

```@docs
gate_to_tensor
```

When $H = \sum_i H_{i,i+1}$, where $H_{i,i+1}$ is a nearest neighbor term, the Trotter gate network naturally forms a square lattice. Each Trotter gate $g_{i,i+1} = \exp(-\delta\tau \, H_{i,i+1})$ connects two spatial sites (horizontal direction in the gate diagram) and advances by one imaginary-time step $\delta\tau$ (vertical direction in the gate diagram). However, the gate network is **rotated by 45 degrees** relative to the standard upright orientation.

[`gate_to_tensor`](@ref) transforms the rotated gate network into an **upright** square tensor network. The resulting tensor $T$ is the elementary cell of this upright network — it lives in the space $V \otimes V \leftarrow V \otimes V$ and follows the standard TNRKit leg convention:

```
    3
    ↓
1 ← T ← 4
    ↓
    2
```

The horizontal direction (legs 1 ↔ 4) corresponds to the spatial dimension of the quantum chain, and the vertical direction (legs 2 ↔ 3) corresponds to imaginary time. Crucially, the imaginary-time step represented by one $T$ tensor is the **same** $\delta\tau$ as one Trotter gate — the transformation does not coarse-grain in time.

## Vertical Stacking

To build up the imaginary-time direction, multiple copies of $T$ are stacked vertically. As the stack grows, the bond dimension in the spatial direction would grow exponentially if left unchecked. TNRKit provides two stacking strategies that use HOTRG-style compression to keep the bond dimension under control.

- [`vertical_stack_exp`](@ref) stacks $2^{\text{nfold}}$ copies of $T$ by repeatedly doubling the current stack. Each iteration takes the current stack (representing $2^k$ Trotter steps) and compresses it on top of itself, yielding a stack of $2^{k+1}$ steps. After `nfold` iterations, the result represents $2^{\text{nfold}}$ Trotter steps. This is expected to be the more accurate approach, since only $m$ iterations are needed to reach $2^m$ steps.

- [`vertical_stack_linear`](@ref) stacks exactly $n$ copies of $T$ linearly (as opposed to a power of 2). Each iteration adds one more copy of the original tensor $T$ on top of the current stack. This is useful when you need a specific number of Trotter steps rather than a power of 2.

## Implementing Custom Quantum Models

To construct the partition function of a custom $(1+1)$D quantum Hamiltonian with nearest-neighbor interactions, follow this recipe:

1. **Construct the Trotter gate**: Build a 4-leg tensor representing $\exp(-\delta\tau \, h_{i,i+1})$ where $h_{i,i+1}$ is the nearest-neighbor Hamiltonian term. The gate should live in $V \otimes V \leftarrow V \otimes V$, where $V$ is the local Hilbert space.

2. **Convert to partition function tensor**: Call [`gate_to_tensor`](@ref)`(gate; trunc=...)` to obtain the elementary tensor $T$.

3. **Stack in the time direction**: Use [`vertical_stack_exp`](@ref) or [`vertical_stack_linear`](@ref) to build up the desired number of Trotter steps.

Now you are ready to feed the resulting 2D tensor network into any of TNRKit's 2D schemes (TRG, HOTRG, LoopTNR, etc.).
