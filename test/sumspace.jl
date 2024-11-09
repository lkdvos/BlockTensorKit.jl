using Test, TestExtras
using TensorKit

@testset "CartesianSpace" begin
    using TensorKit, BlockTensorKit
    using Test, TestExtras

    using TensorKit: hassector

    ds = [2, 3, 2]
    d = sum(ds)

    V = SumSpace(Ref(ℝ) .^ ds)
    # TODO: cannot implement this because of how TensorKit does ⊕
    # @test eval(Meta.parse(sprint(show, V))) == V
    # @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test !isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test V ==
        @constinferred(dual(V)) ==
        @constinferred(conj(V)) ==
        @constinferred(adjoint(V))
    @test field(V) == ℝ

    @test @constinferred(sectortype(V)) == Trivial
    @test ((@constinferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @constinferred(hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    W = @constinferred SumSpace(ℝ^1)
    @test @constinferred(oneunit(V)) == W == oneunit(typeof(V))
    @test @constinferred(⊕(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊕(V, oneunit(V))) == SumSpace(vcat(V.spaces, ℝ^1))
    @test @constinferred(⊕(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(V, V)) ≅ SumSpace(ℝ^(d^2))
    @test @constinferred(fuse(V, V', V, V')) ≅ SumSpace(ℝ^(d^4))
    @test @constinferred(flip(V)) ≅ V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))
end

@testset "ComplexSpace" begin
    using TensorKit, BlockTensorKit
    using Test, TestExtras

    using TensorKit: hassector

    ds = [2, 3, 2]
    d = sum(ds)
    V = SumSpace(ComplexSpace.(ds))

    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V))
    @test field(V) == ℂ

    @test @constinferred(sectortype(V)) == Trivial
    @test ((@constinferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @constinferred(hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    W = @constinferred SumSpace(ℂ^1)
    @test @constinferred(oneunit(V)) == W == oneunit(typeof(V))
    @test @constinferred(⊕(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊕(V, oneunit(V))) == SumSpace(vcat(V.spaces, ℂ^1))
    @test @constinferred(⊕(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(V, V)) ≅ SumSpace(ℂ^(d^2))
    @test @constinferred(fuse(V, V', V, V')) ≅ SumSpace(ℂ^(d^4))
    @test @constinferred(flip(V)) ≅ V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))
end

@testset"GradedSpace" begin
    using TensorKit, BlockTensorKit
    using Test, TestExtras

    using TensorKit: hassector

    V1 = U1Space(0 => 1, 1 => 1)
    V2 = U1Space(0 => 1, 1 => 2)
    V3 = U1Space(0 => 1, 1 => 1)
    d = dim(V1) + dim(V2) + dim(V3)
    V = SumSpace(V1, V2, V3)

    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V))
    @test field(V) == ℂ

    @test @constinferred(sectortype(V)) == sectortype(V1)
    @test ((@constinferred sectors(V))...,) == (U1Irrep(0), U1Irrep(1))
    @test length(sectors(V)) == 2
    @test @constinferred(hassector(V, U1Irrep(0)))
    @test !@constinferred(hassector(V, U1Irrep(2)))
    @test @constinferred(dim(V)) ==
        d ==
        @constinferred(dim(V, U1Irrep(0))) + @constinferred(dim(V, U1Irrep(1)))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    W = @constinferred SumSpace(U1Space(0 => 1))
    @test @constinferred(oneunit(V)) == W == @constinferred(oneunit(typeof(V)))
    @test @constinferred(⊕(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊕(V, oneunit(V))) == SumSpace(vcat(V.spaces, oneunit(V1)))
    @test @constinferred(⊕(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(V, V)) ≅ SumSpace(U1Space(0 => 9, 1 => 24, 2 => 16))
    @test @constinferred(fuse(V, V', V, V')) ≅
        SumSpace(U1Space(0 => 913, 1 => 600, -1 => 600, 2 => 144, -2 => 144))
    @test @constinferred(flip(V)) ≅ SumSpace(flip.(V.spaces)...)
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))
end
