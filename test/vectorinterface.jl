using Test, TestExtras
using TensorKit
using BlockTensorKit
using VectorInterface

Vtr = (SumSpace(ℂ^2, ℂ^1), SumSpace(ℂ^4)', SumSpace(ℂ^2, ℂ^2, ℂ^1), SumSpace(ℂ^2, ℂ^2, ℂ^2),
       SumSpace(ℂ^2, ℂ^2, ℂ^3)')

for V in (Vtr,), k in rand(0:5, 3)
    W = prod(V[1:k]; init=one(V[1])) ← prod(V[(k + 1):end]; init=one(V[1]))
    @testset "Trivial ($T)" for T in (Float64, ComplexF64)
        t = rand(T, W)
        @test T == @constinferred scalartype(T)

        @test @constinferred(norm(t)) isa real(T)
        @test norm(t)^2 ≈ inner(t, t)

        α = rand(T)
        @test (α * α) * t ≈ α * (α * t)
        @test norm(α * t) ≈ abs(α) * norm(t)
        @test t + t ≈ 2 * t
        @test t - t ≈ zero(t)
        @test -t ≈ -one(T) * t

        t2 = rand(T, W)
        β = rand(T)
        @test inner(β * t2, α * t) ≈ conj(β) * α * conj(inner(t, t2))
        @test inner(t, t2) ≈ conj(inner(t2, t))
    end
end
