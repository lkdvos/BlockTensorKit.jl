using Test
using TestExtras
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
W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
W_empty = V1 ⊗ V2 ← eltype(V1)()

const leftorth_algs = (
    TensorKit.QR(),
    TensorKit.QRpos(),
    TensorKit.QL(),
    TensorKit.QLpos(),
    TensorKit.Polar(),
    TensorKit.SVD(),
    TensorKit.SDD(),
)

@testset "leftorth with $alg" for alg in leftorth_algs
    for T in (Float32, ComplexF64), isadjoint in (false, true)
        t = isadjoint ? rand(T, W)' : rand(T, W)
        Q, R = @inferred leftorth(t, ((3, 4, 2), (1, 5)); alg)
        QdQ = Q' * Q
        @test QdQ ≈ one(QdQ)
        @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))
        if alg isa Polar
            @test isposdef(R)
            @test domain(R) == codomain(R) == space(t, 1)' ⊗ space(t, 5)'
        end

        t_empty = isadjoint ? rand(T, W_empty')' : rand(T, W_empty)
        Q, R = @constinferred leftorth(t_empty; alg)
        @test Q == t_empty
        @test dim(Q) == dim(R) == 0
    end
end

const leftnull_algs = (TensorKit.QR(), TensorKit.SVD(), TensorKit.SDD())
@testset "leftnull with $alg" for alg in leftnull_algs
    for T in (Float32, ComplexF64), isadjoint in (false, true)
        t = isadjoint ? rand(T, W)' : rand(T, W)
        N = @constinferred leftnull(t, ((3, 4, 2), (1, 5)); alg=alg)
        NdN = N' * N
        @test NdN ≈ one(NdN)
        @test norm(N' * permute(t, ((3, 4, 2), (1, 5)))) < 100 * eps(norm(t))

        t_empty = isadjoint ? rand(T, W_empty')' : rand(T, W_empty)
        N = @constinferred leftnull(t_empty; alg=alg)
        @test N' * N ≈ id(domain(N))
        @test N * N' ≈ id(codomain(N))
    end
end

const rightorth_algs = (
    TensorKit.RQ(),
    TensorKit.RQpos(),
    TensorKit.LQ(),
    TensorKit.LQpos(),
    TensorKit.Polar(),
    TensorKit.SVD(),
    TensorKit.SDD(),
)
@testset "rightorth with $alg" for alg in rightorth_algs
    for T in (Float32, ComplexF64), isadjoint in (false, true)
        t = isadjoint ? rand(T, W)' : rand(T, W)
        L, Q = @constinferred rightorth(t, ((3, 4), (2, 1, 5)); alg=alg)
        QQd = Q * Q'
        @test QQd ≈ one(QQd)
        @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))
        if alg isa Polar
            @test isposdef(L)
            @test domain(L) == codomain(L) == space(t, 3) ⊗ space(t, 4)
        end

        t_empty = isadjoint ? rand(T, W_empty)' : rand(T, W_empty')
        L, Q = @constinferred rightorth(t_empty; alg=alg)
        @test Q == t_empty
        @test dim(Q) == dim(L) == 0
    end
end

const rightnull_algs = (TensorKit.LQ(), TensorKit.SVD(), TensorKit.SDD())
@testset "rightnull with $alg" for alg in rightnull_algs
    for T in (Float32, ComplexF64), isadjoint in (false, true)
        t = isadjoint ? rand(T, W)' : rand(T, W)
        M = @constinferred rightnull(t, ((3, 4), (2, 1, 5)); alg=alg)
        MMd = M * M'
        @test MMd ≈ one(MMd)
        @test norm(permute(t, ((3, 4), (2, 1, 5))) * M') < 100 * eps(norm(t))

        t_empty = isadjoint ? rand(T, W_empty)' : rand(T, W_empty')
        M = @constinferred rightnull(t_empty; alg)
        @test M * M' ≈ id(codomain(M))
        @test M' * M ≈ id(domain(M))
    end
end

const svd_algs = (TensorKit.SVD(), TensorKit.SDD())
@testset "tsvd with $alg" for alg in svd_algs
    for T in (Float32, ComplexF64), isadjoint in (false, true)
        t = isadjoint ? rand(T, W)' : rand(T, W)
        U, S, V = @constinferred tsvd(t, ((3, 4, 2), (1, 5)); alg)
        UdU = U' * U
        @test UdU ≈ one(UdU)
        VVd = V * V'
        @test VVd ≈ one(VVd)
        @test U * S * V ≈ permute(t, ((3, 4, 2), (1, 5)))

        t_empty = isadjoint ? rand(T, W_empty')' : rand(T, W_empty)
        U, S, V = @inferred tsvd(t_empty; alg=alg)
        @test U == t_empty
        @test dim(U) == dim(S) == dim(V)
    end
end

#         t = Tensor(rand, T, V1 ⊗ V1' ⊗ V2 ⊗ V2')
#         @testset "eig and isposdef" begin
#             D, V = eigen(t, (1, 3), (2, 4))
#             D̃, Ṽ = @constinferred eig(t, (1, 3), (2, 4))
#             @test D ≈ D̃
#             @test V ≈ Ṽ
#             VdV = V' * V
#             VdV = (VdV + VdV') / 2
#             @test isposdef(VdV)
#             t2 = permute(t, (1, 3), (2, 4))
#             @test t2 * V ≈ V * D
#             @test !isposdef(t2) # unlikely for non-hermitian map
#             t2 = (t2 + t2')
#             D, V = eigen(t2)
#             VdV = V' * V
#             @test VdV ≈ one(VdV)
#             D̃, Ṽ = @constinferred eigh(t2)
#             @test D ≈ D̃
#             @test V ≈ Ṽ
#             λ = minimum(minimum(real(LinearAlgebra.diag(b))) for (c, b) in blocks(D))
#             @test isposdef(t2) == isposdef(λ)
#             @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
#             @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))
#         end
#     end
# end
# @timedtestset "Tensor truncation" begin
#     for T in (Float32, ComplexF64)
#         for p in (1, 2, 3, Inf)
#             # Test both a normal tensor and an adjoint one.
#             ts = (TensorMap(randn, T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
#                   TensorMap(randn, T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)')
#             for t in ts
#                 U₀, S₀, V₀, = tsvd(t)
#                 t = rmul!(t, 1 / norm(S₀, p))
#                 U, S, V, ϵ = @constinferred tsvd(t; trunc=truncerr(5e-1), p=p)
#                 # @show p, ϵ
#                 # @show domain(S)
#                 # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncerr(nextfloat(ϵ)), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncdim(ceil(Int, dim(domain(S)))), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#                 # results with truncationcutoff cannot be compared because they don't take degeneracy into account, and thus truncate differently
#                 U, S, V, ϵ = tsvd(t; trunc=truncbelow(1 / dim(domain(S₀))), p=p)
#                 # @show p, ϵ
#                 # @show domain(S)
#                 # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
#                 U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
#                 @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
#             end
#         end
#     end
# end
