@testset "Q-system property of FunZN ∈ Rep[DN]" begin
    for N in 2:7
        FunZN, m = TNRKit.FunZN_Dihedral(N)
        @tensor AA_A[1; 2 3 4] := m[middle; 2 3] * m[1; middle 4]
        @tensor A_AA[1; 2 3 4] := m[middle; 3 4] * m[1; 2 middle]

        @test AA_A ≈ A_AA # Associativity is satisfied

        @test m * m' ≈ id(FunZN) # Isometry is satisfied
    end
end
