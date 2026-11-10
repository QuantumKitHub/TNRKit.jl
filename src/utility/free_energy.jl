"""
$(SIGNATURES)

Takes a vector of normalization factors `data` and a given inverse temperature `β`
and computes the free energy.

!!! info
    The `scalefactor` should be set to the rescaling factor of the area of the tensor network after each iteration of the TNR algorithm.

    The `initial_size` should be set to the intial size of the physical lattice, which is typically `1.0`.

"""
function free_energy(data, β; scalefactor = 2.0, initial_size = 1.0)
    lnz = 0.0
    x = 1.0 - log(initial_size) / log(scalefactor)
    for (i, z) in enumerate(data)
        lnz += log(z) * scalefactor^(x - i)
    end
    return -lnz / β
end
