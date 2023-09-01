using BlockTensorKit
using Test, TestExtras
using Random
using TensorKit
using LinearAlgebra
using VectorInterface
using Combinatorics
Random.seed!(12345)

V = (SumSpace(ℂ^2, ℂ^1), SumSpace(ℂ^2))
V1, V2 = V

t = Tensor((T, d) -> rand(-3:3, d...), Int, V1 ⊗ V2)
rmul!(t[2], 100)

a = convert(TensorMap, t)

k = 1
p = (2, 1)
p1 = ntuple(n -> p[n], k)
p2 = ntuple(n -> p[k + n], length(V) - k)
t2 = permute(t, p1, p2; copy=true)
a2 = convert(TensorMap, t2)
@test a2 ≈ permute(a, p1, p2; copy=true)
