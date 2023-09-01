using BlockTensorKit
using Test, TestExtras
using Random
using TensorKit
using LinearAlgebra
using VectorInterface
using Combinatorics
Random.seed!(12345)

@testset "BlockTensorKit.jl" verbose=true begin
    # include("sumspace.jl")
    # include("linalg.jl")
    include("tensorops.jl")
end
