using TensorKit, BlockTensorKit
using Test, TestExtras
using Random

T = Float64
dims = ([2, 3], [3, 2], [1, 2], [2, 1], [2])
spaces = map(ds -> SumSpace(CartesianSpace.(ds)), dims)

bA = TensorMap(randn, T, ⊗(spaces[1:2]...), ⊗(spaces[3:5]...))
tA = convert(TensorMap, bA)

@test (norm(bA)) ≈ norm(tA)

bB = TensorMap(randn, T, ⊗(spaces[1:2]...), ⊗(spaces[3:5]...))

tB = convert(TensorMap, bB)
α = randn(T)
t = bA
@test @constinferred(2 * bA) ≈ @constinferred(bA + bA)
@test (2.0 * bA) / 2.0 ≈ bA

@test convert(TensorMap, @constinferred(α * bA)) ≈ α * tA
@test convert(TensorMap, @constinferred(bA + bB)) ≈ tA + tB
