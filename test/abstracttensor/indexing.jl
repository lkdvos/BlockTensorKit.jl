using BlockTensorKit
using BlockTensorKit: sprand
using Test
using TensorKit

V = SumSpace(ℂ^2, ℂ^3, ℂ^2)

blockt = rand(V ⊗ V ⊗ V)

# scalar indexing
@test @inferred(blockt[1]) isa TensorMap
@test @inferred(blockt[1, 1, 1]) isa TensorMap
@test @inferred(blockt[CartesianIndex(1, 1, 1)]) isa TensorMap

# colon indexing
blockt2 = @inferred blockt[:, :, :]
@test blockt2 == blockt
for I in eachindex(blockt)
    @test blockt[I] === blockt[I]
end

@test size(@inferred(blockt[1, :, 1])) == (1, 3, 1)
blockt2 = blockt[1, [1, 3], 1]
@test size(blockt2) == (1, 2, 1)
blockt3 = @inferred blockt[[1], [1], 1]
@test blockt3 isa BlockTensorMap
@test length(blockt3) == 1

# invalid indexing
@test_throws MethodError blockt[:]
@test_throws MethodError blockt[[1]]

blockt = sprand(V ⊗ V ⊗ V, 0.5)

# scalar indexing
@test @inferred(blockt[1]) isa TensorMap
@test @inferred(blockt[1, 1, 1]) isa TensorMap
@test @inferred(blockt[CartesianIndex(1, 1, 1)]) isa TensorMap

# colon indexing
blockt2 = @inferred blockt[:, :, :]
@test blockt2 == blockt
nnz = nonzero_length(blockt)
for I in eachindex(blockt)
    if I in nonzero_keys(blockt)
        @test blockt[I] === blockt[I]
    else
        @test norm(blockt[I]) == 0
        @test nonzero_length(blockt) == nnz
    end
end

@test size(@inferred(blockt[1, :, 1])) == (1, 3, 1)
blockt2 = blockt[1, [1, 3], 1]
@test size(blockt2) == (1, 2, 1)
blockt3 = @inferred blockt[[1], [1], 1]
@test blockt3 isa SparseBlockTensorMap
@test length(blockt3) == 1

# invalid indexing
@test_throws MethodError blockt[:]
@test_throws MethodError blockt[[1]]
