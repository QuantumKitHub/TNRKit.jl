println("--------")
println(" spaces ")
println("--------")

# stuff different kinds of spaces into the schemes (and do 25 steps)
A = ("Normal Ising", classical_ising(Trivial))
B = ("Ising Symmetric", classical_ising())
C = ("Gross Neveu", gross_neveu_start(0, 1, 0))
D = ("Sixvertex U1", sixvertex(U1Irrep; T = ComplexF64))

models = [A, B, C, D]
schemes = [TRG, BTRG, HOTRG, ATRG]

# The tests below check that the schemes don't lead to spacemismatches
for S in schemes
    @testset "$(S) - spaces" begin
        for T in models
            @eval begin
                println("--- $($(T[1])) ---")
                @test isa(run!($S($(T[2])), truncrank(7), maxiter(25)), Any)
            end
        end
    end
end
