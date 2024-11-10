
using Test
using TensorKit
using BlockTensorKit
using Random
using Combinatorics

Vtr = (
    SumSpace(ℂ^3),
    SumSpace(ℂ^2, ℂ^2)',
    SumSpace(ℂ^2, ℂ^2, ℂ^1),
    SumSpace(ℂ^2, ℂ^2, ℂ^2),
    SumSpace(ℂ^2, ℂ^3, ℂ^1, ℂ^1)',
)

for V in (Vtr,)
    V1, V2, V3, V4, V5 = V
    @assert V3 * V4 * V2 ≿ V1' * V5' # necessary for leftorth tests
    @assert V3 * V4 ≾ V1' * V2' * V5' # necessary for rightorth tests
end

spacelist = (Vtr,)
scalartypes = (Float64, ComplexF64)
V = first(spacelist)
# @testset "Tensors with symmetry: $(TensorKit.type_repr(sectortype(first(V))))" verbose = true failfast=true for V in
#   spacelist
I = sectortype(first(V))
V1, V2, V3, V4, V5 = V

@testset "Basic tensor properties" begin
    W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
    for T in scalartypes
        t = @inferred sparse(zeros(T, W))
        @test t isa SparseBlockTensorMap
        @test @inferred(hash(t)) == hash(deepcopy(t))
        @test scalartype(t) == T
        @test iszero(norm(t))
        @test W == @inferred codomain(t)
        @test one(W) == @inferred domain(t)
        @test (W ← one(W)) == @inferred space(t)
    end
end

@testset "TensorMap conversion" begin
    W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
    for T in scalartypes
        t1 = sprand(T, W, 0.5)
        t2 = sprand(T, W, 0.5)
        t1′ = @inferred convert(TensorMap, t1)
        t2′ = @inferred convert(TensorMap, t2)
        @test norm(t1) ≈ norm(t1′)
        @test norm(t2) ≈ norm(t2′)
        @test inner(t1, t2) ≈ inner(t1′, t2′)
        t1″ = @inferred SparseBlockTensorMap(t1′, W)
        t2″ = @inferred SparseBlockTensorMap(t2′, W)
        @test t1 ≈ t1″
        @test t2 ≈ t2″
    end
end
