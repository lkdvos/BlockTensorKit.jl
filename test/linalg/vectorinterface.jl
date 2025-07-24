using Test
using TensorKit
using BlockTensorKit
using BlockTensorKit: sprand
using VectorInterface

Vtr = (
    SumSpace(ℂ^2, ℂ^1),
    SumSpace(ℂ^4)',
    SumSpace(ℂ^2, ℂ^2, ℂ^1),
    SumSpace(ℂ^2, ℂ^2, ℂ^2),
    SumSpace(ℂ^2, ℂ^2, ℂ^3)',
)

V = Vtr

@testset "VectorInterface $(issparse ? "SparseBlockTensorMap" : "BlockTensorMap")" for issparse in
    (
        false, true,
    )
    if issparse
        t = sprand(Float64, *(V[1:3]...) ← *(V[4], V[5]), 0.5)
        t′ = sprand(Float64, *(V[1:3]...) ← *(V[4], V[5]), 0.5)
    else
        t = rand(*(V[1:3]...) ← *(V[4], V[5]))
        t′ = rand(*(V[1:3]...) ← *(V[4], V[5]))
    end

    @testset "scalartype" begin
        @test Float64 === @inferred scalartype(t)
    end

    @testset "zerovector" begin
        tmp = deepcopy(t)
        t1 = @inferred zerovector(tmp)
        t2 = @inferred zerovector!(tmp)
        t3 = @inferred zerovector!!(tmp)

        @test norm(t1) == norm(t2) == norm(t3) == 0
        @test t1 !== tmp
        @test t2 === t3 === tmp

        t4 = @inferred zerovector(t, ComplexF64)
        @test t4 !== t
        @test scalartype(t4) === ComplexF64
        @test iszero(norm(t4))

        t5 = @inferred zerovector!!(tmp, Float64)
        t6 = @inferred zerovector!!(tmp, ComplexF64)
        @test t5 === tmp
        @test t6 !== tmp
    end

    @testset "scale" begin
        α = randn()
        β = randn()

        t1 = @inferred scale(t, α)
        @test norm(t1) ≈ abs(α) * norm(t)
        @test t1 !== t
        @test scale(t, α * β) ≈ scale(scale(t, α), β)

        t2 = @inferred scale!!(t1, inv(α))
        @test t2 === t1
        @test t2 ≈ t

        t3 = @inferred scale!(t1, α)
        @test t3 === t1
        @test t1 ≈ scale(t, α)

        γ = randn(ComplexF64)
        t4 = zerovector(t)
        t5 = @inferred scale!!(t4, t, γ)
        @test t5 !== t
        @test t5 !== t4
        @test scalartype(t5) == typeof(γ)
        @test norm(t5) ≈ norm(t) * abs(γ)

        t6 = zerovector(t, ComplexF64)
        t7 = zerovector(t6)
        t8 = @inferred scale!(t6, t, γ)
        t9 = @inferred scale!!(t7, t, γ)
        @test t6 === t8
        @test t7 === t9
        @test t8 == t9 == t5
    end

    @testset "add" begin
        α, β = rand(2)
        γ, δ = rand(ComplexF64, 2)

        t3 = @inferred add(t, t′)
        @test t3 == add(t′, t)

        @test add(t, t) ≈ scale(t, 2)
        @test @inferred add(t, t, α) ≈ scale(t, (1 + α))
        @test @inferred add(t, t, α, β) ≈ scale(t, (α + β))
        @test add(t, t′, α, β) ≈ add(scale(t, β), scale(t′, α))
        @test @inferred add(t, t′, γ, δ) ≈ @inferred add(scale(t, δ), scale(t′, γ))

        tmp = deepcopy(t)
        t2 = @inferred add!(tmp, t′)
        @test t2 ≈ add(t, t′)
        @test t2 === tmp
        t2 = @inferred add!(deepcopy(t), t′, α)
        @test t2 ≈ add(t, t′, α)
        t2 = @inferred add!(deepcopy(t), t′, α, β)
        @test t2 ≈ add(t, t′, α, β)

        tmp = deepcopy(t)
        t2 = @inferred add!!(tmp, t′)
        @test t2 ≈ add(t, t′)
        @test t2 === tmp
        t2 = @inferred add!!(deepcopy(t), t′, α)
        @test t2 ≈ add(t, t′, α)
        t2 = @inferred add!!(deepcopy(t), t′, α, β)
        @test t2 ≈ add(t, t′, α, β)

        tmp = deepcopy(t)
        t3 = @inferred add!!(tmp, t′, γ, One())
        @test t3 !== tmp
        @test scalartype(t3) === ComplexF64
        @test t3 ≈ add(tmp, t′, γ, One())
    end

    @testset "inner" begin
        @test norm(t)^2 ≈ inner(t, t)
        @test inner(t, t′) ≈ conj(inner(t′, t))
        α, β = rand(ComplexF64, 2)
        t″ = scale(t, α)
        t‴ = scale(t′, β)
        @test @inferred inner(t″, t‴) ≈ conj(α) * β * inner(t, t′)
    end

    @testset "general linalg" begin
        α, β = rand(ComplexF64, 2)
        @test (α * α) * t ≈ α * (α * t)
        @test norm(α * t) ≈ abs(α) * norm(t)
        @test t + t ≈ 2 * t
        @test t - t ≈ zero(t)
        @test -t ≈ -one(scalartype(t)) * t
        @test inner(β * t′, α * t) ≈ conj(β) * α * conj(inner(t, t′))
        @test inner(t, t′) ≈ conj(inner(t′, t))
    end
end
