using Revise, TensorKit, TNRKit

# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

# stop when converged or after 50 steps, whichever comes first
stopping_criterion = convcrit(1.0e-16, trg_f) & maxiter(20)

# choose a TensorKit truncation scheme
trunc = truncrank(16) & trunctol(atol = 1.0e-40)


# ---- old interface (other schemes) ----

# initialize the BTRG scheme
scheme = BTRG(classical_ising(1.0), -0.5)

# run the BTRG scheme
data = run!(scheme, trunc, stopping_criterion)

# initialize the HOTRG scheme
scheme = HOTRG(classical_ising(1.0))

# run the HOTRG scheme
data = run!(scheme, trunc, stopping_criterion)

# ---- iterable `Renormalizer` interface ----

# create a pure algorithm config (kwargs with sensible defaults)
alg = TRG(; trunc = trunc, stop = stopping_criterion)
renorm = Renormalizer(alg, classical_ising(1.0))

# run all steps with logging
state, data = run!(renorm; verbosity = 1)
f = free_energy(data, 1.0)

# or, step through manually
renorm = Renormalizer(alg, classical_ising(1.0))
for (state, data) in renorm
    τ0, _ = extract_tau_and_c(state.T; fast = true)
    # each iteration yields the state after one RG step
    # renorm.step tracks how many steps have been completed
end
# renorm.step is the number of completed steps

# or, advance one step at a time
renorm = Renormalizer(alg, classical_ising(1.0))
rgstep!(renorm)  # throws if stop criterion already met
rgstep!(renorm)

# access the current tensor directly from the state
T_final = renorm.state.T
