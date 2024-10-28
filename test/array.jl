using Test, TestExtras
using BlockTensorKit
using TensorKit
using TensorOperations

@testset "SumSpaceIndices" begin
    V = SumSpace(ℂ^2, ℂ^2) ← SumSpace(ℂ^2, ℂ^2)
    eachspace = @constinferred SumSpaceIndices(V)
    
    @test (ℂ^2 ← ℂ^2) == @constinferred eachspace[1]
    @test (ℂ^2 ← ℂ^2) == @constinferred eachspace[1, 2]
    @test SumSpaceIndices(SumSpace(ℂ^2, ℂ^2) ← ℂ^2) == (@constinferred eachspace[:, 1])
end

V = SumSpace(ℂ^2, ℂ^2)
A = BlockTensorMap{Float64}(undef, V ← V)
block(A, Trivial())