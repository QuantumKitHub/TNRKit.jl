function VN_entropy(M::TensorMap; rtol = 1.0e-14, power = 1.0)
    Λ, U = eigen(M)
    Λ_vec_norm = Λ.data / sum(Λ.data)
    plogp = map(x -> norm(x) > rtol ? x^power * power * log(x) : 0.0, Λ_vec_norm)
    S_von = - sum(plogp)
    return S_von, Λ / maximum(abs.(Λ.data))
end

"""
Given a LoopTNR scheme, consider the loop MPS |ψ⟩ = tr(TA * TB * TA * TB). 
Compute a gauge-invariant data called loop entropy on each of the four edges of the loop MPS.
This loop entropy is defined as the von Neumann entropy of the transfer matrix formed by contracting the loop MPS with itself, with the bond of interest as the cut.
The loop entropy is a measure of the entanglement across the bond, and can be used to diagnose the presence of short-range entanglement in the loop MPS.
The entanglement spectrum is also returned.

### References
* [Evenbly et. al. Phys. Rev. B 98 (2018)](@cite Evenbly_2018)

"""
function loop_entropy(scheme::LoopTNR)
    psi_A = Ψ_A(scheme)
    psi_Apsi_A_vector = ΨAΨA(psi_A)
    N = length(psi_A)
    psi_Apsi_A_cache = right_cache(psi_Apsi_A_vector)

    entropies_circ = ComplexF64[]
    specs_circ = DiagonalTensorMap{ComplexF64}[]
    entropies_rad = ComplexF64[]
    specs_rad = DiagonalTensorMap{ComplexF64}[]

    psi_Apsi_A = psi_Apsi_A_cache[end]

    for i in 1:N
        psi_Apsi_A = psi_Apsi_A * psi_Apsi_A_vector[i]
        transfer = psi_Apsi_A_cache[i] * psi_Apsi_A
        ent_circ, spec_circ = VN_entropy(transfer)
        ent_rad, spec_rad = VN_entropy(transpose(transfer, ((2, 4), (1, 3))))
        push!(entropies_circ, ent_circ)
        push!(specs_circ, spec_circ)
        push!(entropies_rad, ent_rad)
        push!(specs_rad, spec_rad)
    end

    return entropies_circ, specs_circ, entropies_rad, specs_rad
end
