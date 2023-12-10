using TensorKit

@testset "CartesianSpace" begin
    ds = [2, 3, 2]
    d = sum(ds)

    V = SumSpace(Ref(ℝ) .^ ds)
    @test eval(Meta.parse(sprint(show, V))) == V
    @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)

    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanProduct)
    @test isa(V, SumSpace)

    @test !isdual(V)
    @test !isdual(V')

    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test V == @constinferred(dual(V)) == @constinferred(conj(V)) ==
          @constinferred(adjoint(V))
    @test field(V) == ℝ

    @test @constinferred(sectortype(V)) == Trivial
    @test ((@constinferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @constinferred(TensorKit.hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test dim(@constinferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @constinferred(TensorKit.axes(V)) == Base.OneTo(d)
    W = @constinferred SumSpace(ℝ^1)
    @test @constinferred(oneunit(V)) == W == oneunit(typeof(V))
    @test @constinferred(⊕(V, V)) == SumSpace(vcat(V.spaces, V.spaces))
    @test @constinferred(⊕(V, oneunit(V))) == SumSpace(vcat(V.spaces, ℝ^1))
    @test @constinferred(⊕(V, V, V, V)) == SumSpace(repeat(V.spaces, 4))
    @test @constinferred(fuse(V, V)) == SumSpace(ℝ^(d^2))
    @test @constinferred(fuse(V, V', V, V')) == SumSpace(ℝ^(d^4))
    @test @constinferred(flip(V)) == V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))
    # @test @constinferred(infimum(V, ℝ^3)) == V
    # @test @constinferred(supremum(V', ℝ^3)) == ℝ^3
end
