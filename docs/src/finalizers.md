# Finalizers
At the end of every TNR step (and before the first step if `finalize_beginning=true` is chose in the `run!` function, which is default behaviour), the state of the scheme is finalized.

By default this finalization process is as follow:

We calculate the "norm" of the scheme's tensor(s) by taking the trace over the lattice directions.
To keep the numbers in the tensor(s) from diverging, we divide the tensor(s) by this norm.

For TRG this is for example:
```Julia
n = norm(@tensor T[1 2; 2 1])
T /= n
```

At the end of a simulation, the `run!` function returns a vector of these norms. You can take this data to calculate the free energy through the `free_energy(data, β)` function for example.

This finalization is handled through what we call [`Finalizer`](@ref)s.

These [`Finalizer`](@ref)s are a way for the user to calculate all sorts of things throughout a TNR calculation.

A custom instance of `Finalizer` can be created as:
```Julia
function my_finalization(scheme::TRG)
    n = finalize!(scheme) # normalizes the tensor and return said norm
    data = calculate_something(scheme)
    return n, data # Two Float64s
end

custom_Finalizer = Finalizer(my_finalization, Tuple{Float64, Float64})
```

And can then by used by setting the `finalizer` kwarg in the `run!` function:
```Julia
data = run!(...; finalizer=custom_Finalizer)
```

A `Finalizer` has 1 field `f!` which is the function being called on the scheme (`f!(scheme)`) at the time of finalization. It also has a type parameter `E` that corresponds to the output type of `f!`.
We use this type parameter `E` to correctly allocate a `Vector{E}` in which all the data will be stored throughout the simulation.

## Examples

### Built-in Finalizers

The default [`Finalizer`](@ref) is `default_Finalizer` which normalizes the tensor(s) and stores the norm.
For the impurity methods ([`ImpurityTRG`](@ref) and [`ImpurityHOTRG`](@ref)) the defaults are `ImpurityTRG_Finalizer` and `ImpurityHOTRG_Finalizer` respectively, as these methods usually require us to store more than just one norm per iteration.

[`TRG`](@ref), [`ATRG`](@ref), [`HOTRG`](@ref) and [`BTRG`](@ref) can be normalized by calculating the norm of a 2x2 patch of tensors, which is more computationally expensive but should™ be more stable.

TNRKit exports the following pre-built `Finalizer` instances:

- **`two_by_two_Finalizer`** - Normalizes using a 2×2 patch of tensors (more stable but computationally more expensive). Works with [`TRG`](@ref), [`ATRG`](@ref), [`HOTRG`](@ref), and [`BTRG`](@ref).

- **`GSDegeneracy_Finalizer`** - Computes the ground state degeneracy at each TNR step. Returns a `Float64` at each iteration.

- **`guwenratio_Finalizer`** - Computes the Gu-Wen ratio. Returns a `Tuple{Float64, Float64}` at each iteration.

### Usage Examples

```julia
using TNRKit

# Default finalization (simple norm)
T = classical_ising(ising_βc)
scheme = TRG(T)
data = run!(scheme, truncrank(16), maxiter(25))

# Use the two-by-two normalizer (more stable)
T = classical_ising(ising_βc)
scheme = TRG(T)
data = run!(scheme, truncrank(16), maxiter(25); finalizer=two_by_two_Finalizer)

# Track ground state degeneracy throughout the simulation
T = classical_ising(ising_βc)
scheme = TRG(T)
gsd_data = run!(scheme, truncrank(16), maxiter(25); finalizer=GSDegeneracy_Finalizer)
```