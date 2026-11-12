function block_tensors(O_list::Matrix)
    m, n = size(O_list)
    ind_list = [
        [
                (n + 1) * (i - 1) + j,
                m * (n + 1) + (m + 1) * (j - 1) + i + 1,
                m * (n + 1) + (m + 1) * (j - 1) + i,
                (n + 1) * (i - 1) + j + 1,
            ] for i in 1:m for j in 1:n
    ]
    Vv = [space(O_list[end, j])[2] for j in 1:n]
    Uv = isomorphism(fuse(Vv...), prod(Vv))
    Vh = [space(O_list[i, 1])[1] for i in 1:m]
    Uh = isomorphism(fuse(Vh...), prod(Vh))

    ind1 = vcat(-1, [(n + 1) * (i - 1) + 1 for i in 1:m])
    ind2 = vcat(-2, [m * (n + 1) + (m + 1) * j for j in 1:n])
    ind3 = vcat([m * (n + 1) + (m + 1) * (j - 1) + 1 for j in 1:n], -3)
    ind4 = vcat([(n + 1) * i for i in 1:m], -4)

    tensors = vcat(O_list[:], [Uh, Uv, adjoint(Uv), adjoint(Uh)])

    inds = vcat(ind_list, [ind1, ind2, ind3, ind4])
    return permute(ncon(tensors, inds), ((1, 2), (3, 4)))
end
