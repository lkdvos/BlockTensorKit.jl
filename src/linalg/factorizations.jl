using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, YALAPACK.BlasMat, Algorithm
import MatrixAlgebraKit as MAK

# Type piracy for defining the MAK rules on BlockArrays!
# -----------------------------------------------------

MAK.@algdef BlockAlgorithm

const BlockBlasMat{T <: MAK.BlasFloat} = BlockMatrix{T}

for f in (
        :svd_compact, :svd_full, :svd_trunc, :svd_vals, :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null, :eig_full, :eig_trunc, :eig_vals, :eigh_full,
        :eigh_trunc, :eigh_vals, :left_polar, :right_polar,
    )
    f! = Symbol(f, :!)

    @eval MAK.copy_input(::typeof($f), A::BlockBlasMat) = Array(A)
    @eval MAK.$f!(t::BlockBlasMat, F, alg::BlockAlgorithm) = $f(t; alg.kwargs...)
end

for f in (:qr, :lq, :eig, :eigh, :gen_eig, :svd, :polar)
    default_f_algorithm = Symbol(:default_, f, :_algorithm)
    @eval MAK.$default_f_algorithm(::Type{<:BlockBlasMat{T}}; kwargs...) where {T} =
        BlockAlgorithm(; kwargs...)
end

# Make sure sparse blocktensormaps have dense outputs
function MAK.initialize_output(::typeof(qr_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(codomain(t))
    Q = dense_similar(t, codomain(t) ← V_Q)
    R = dense_similar(t, V_Q ← domain(t))
    return Q, R
end
function MAK.initialize_output(::typeof(qr_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    Q = dense_similar(t, codomain(t) ← V_Q)
    R = dense_similar(t, V_Q ← domain(t))
    return Q, R
end
function MAK.initialize_output(::typeof(qr_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = dense_similar(t, codomain(t) ← V_N)
    return N
end

function MAK.initialize_output(::typeof(lq_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(domain(t))
    L = dense_similar(t, codomain(t) ← V_Q)
    Q = dense_similar(t, V_Q ← domain(t))
    return L, Q
end
function MAK.initialize_output(::typeof(lq_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    L = dense_similar(t, codomain(t) ← V_Q)
    Q = dense_similar(t, V_Q ← domain(t))
    return L, Q
end
function MAK.initialize_output(::typeof(lq_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    N = dense_similar(t, V_N ← domain(t))
    return N
end

function MAK.initialize_output(::typeof(left_polar!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    W = dense_similar(t, space(t))
    P = dense_similar(t, domain(t) ← domain(t))
    return W, P
end
function MAK.initialize_output(::typeof(right_polar!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    P = dense_similar(t, codomain(t) ← codomain(t))
    Wᴴ = dense_similar(t, space(t))
    return P, Wᴴ
end

function MAK.initialize_output(::typeof(left_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    return dense_similar(t, codomain(t) ← V_N)
end
function MAK.initialize_output(::typeof(right_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    return dense_similar(t, V_N ← domain(t))
end

function MAK.initialize_output(::typeof(eigh_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = dense_similar(t, codomain(t) ← V_D)
    return D, V
end
function MAK.initialize_output(::typeof(eig_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = dense_similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(svd_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = dense_similar(t, codomain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = dense_similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end
function MAK.initialize_output(::typeof(svd_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = dense_similar(t, codomain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
    Vᴴ = dense_similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

# function TK.leftorth!(
#         t::AbstractBlockTensorMap;
#         alg::Union{QR, QRpos, QL, QLpos, SVD, SDD, Polar} = QRpos(),
#         atol::Real = zero(float(real(scalartype(t)))),
#         rtol::Real = if (alg ∉ (SVD(), SDD()))
#             zero(float(real(scalartype(t))))
#         else
#             eps(real(float(one(scalartype(t))))) * iszero(atol)
#         end,
#     )
#     InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:leftorth!)
#     if !iszero(rtol)
#         atol = max(atol, rtol * norm(t))
#     end
#     I = sectortype(t)
#     dims = TK.SectorDict{I, Int}()
#
#     # compute QR factorization for each block
#     if !isempty(TK.blocks(t))
#         generator = Base.Iterators.map(TK.blocks(t)) do (c, b)
#             Qc, Rc = TK.MatrixAlgebra.leftorth!(b, alg, atol)
#             dims[c] = size(Qc, 2)
#             return c => (Qc, Rc)
#         end
#         QRdata = SectorDict(generator)
#     end
#
#     # construct new space
#     S = spacetype(t)
#     V = S(dims)
#     if alg isa Polar
#         @assert V ≅ domain(t)
#         W = domain(t)
#     else
#         W = ProductSpace(V)
#     end
#
#     # construct output tensors
#     T = float(scalartype(t))
#     Q = similar(t, T, codomain(t) ← W)
#     R = similar(t, T, W ← domain(t))
#     if !isempty(blocksectors(domain(t)))
#         for (c, (Qc, Rc)) in QRdata
#             block(Q, c) .= Qc
#             block(R, c) .= Rc
#         end
#     end
#     return Q, R
# end
# function TK.leftorth!(t::SparseBlockTensorMap; kwargs...)
#     return leftorth!(BlockTensorMap(t); kwargs...)
# end
#
# function TK.leftnull!(
#         t::BlockTensorMap;
#         alg::Union{QR, QRpos, SVD, SDD} = QRpos(),
#         atol::Real = zero(float(real(scalartype(t)))),
#         rtol::Real = if (alg ∉ (SVD(), SDD()))
#             zero(float(real(scalartype(t))))
#         else
#             eps(real(float(one(scalartype(t))))) * iszero(atol)
#         end,
#     )
#     InnerProductStyle(t) === EuclideanInnerProduct() ||
#         throw_invalid_innerproduct(:leftnull!)
#     if !iszero(rtol)
#         atol = max(atol, rtol * norm(t))
#     end
#     I = sectortype(t)
#     dims = SectorDict{I, Int}()
#
#     # compute QR factorization for each block
#     V = codomain(t)
#     if !isempty(blocksectors(V))
#         generator = Base.Iterators.map(blocksectors(V)) do c
#             Nc = MatrixAlgebra.leftnull!(block(t, c), alg, atol)
#             dims[c] = size(Nc, 2)
#             return c => Nc
#         end
#         Ndata = SectorDict(generator)
#     end
#
#     # construct new space
#     S = spacetype(t)
#     W = S(dims)
#
#     # construct output tensor
#     T = float(scalartype(t))
#     N = similar(t, T, V ← W)
#     if !isempty(blocksectors(V))
#         for (c, Nc) in Ndata
#             copy!(block(N, c), Nc)
#         end
#     end
#     return N
# end
# TK.leftnull!(t::SparseBlockTensorMap; kwargs...) = leftnull!(BlockTensorMap(t); kwargs...)
#
# function TK.rightorth!(
#         t::AbstractBlockTensorMap;
#         alg::Union{LQ, LQpos, RQ, RQpos, SVD, SDD, Polar} = LQpos(),
#         atol::Real = zero(float(real(scalartype(t)))),
#         rtol::Real = if (alg ∉ (SVD(), SDD()))
#             zero(float(real(scalartype(t))))
#         else
#             eps(real(float(one(scalartype(t))))) * iszero(atol)
#         end,
#     )
#     InnerProductStyle(t) === EuclideanInnerProduct() ||
#         throw_invalid_innerproduct(:rightorth!)
#     if !iszero(rtol)
#         atol = max(atol, rtol * norm(t))
#     end
#     I = sectortype(t)
#     dims = TK.SectorDict{I, Int}()
#
#     # compute LQ factorization for each block
#     if !isempty(TK.blocks(t))
#         generator = Base.Iterators.map(TK.blocks(t)) do (c, b)
#             Lc, Qc = TK.MatrixAlgebra.rightorth!(b, alg, atol)
#             dims[c] = size(Qc, 1)
#             return c => (Lc, Qc)
#         end
#         LQdata = SectorDict(generator)
#     end
#
#     # construct new space
#     S = spacetype(t)
#     V = S(dims)
#     if alg isa Polar
#         @assert V ≅ codomain(t)
#         W = codomain(t)
#     else
#         W = ProductSpace(V)
#     end
#
#     # construct output tensors
#     T = float(scalartype(t))
#     L = similar(t, T, codomain(t) ← W)
#     Q = similar(t, T, W ← domain(t))
#     if !isempty(blocksectors(codomain(t)))
#         for (c, (Lc, Qc)) in LQdata
#             block(L, c) .= Lc
#             block(Q, c) .= Qc
#         end
#     end
#     return L, Q
# end
# function TK.rightorth!(t::SparseBlockTensorMap; kwargs...)
#     return rightorth!(BlockTensorMap(t); kwargs...)
# end
#
# function TK.rightnull!(
#         t::BlockTensorMap;
#         alg::Union{LQ, LQpos, SVD, SDD} = LQpos(),
#         atol::Real = zero(float(real(scalartype(t)))),
#         rtol::Real = if (alg ∉ (SVD(), SDD()))
#             zero(float(real(scalartype(t))))
#         else
#             eps(real(float(one(scalartype(t))))) * iszero(atol)
#         end,
#     )
#     InnerProductStyle(t) === EuclideanInnerProduct() ||
#         throw_invalid_innerproduct(:rightnull!)
#     if !iszero(rtol)
#         atol = max(atol, rtol * norm(t))
#     end
#     I = sectortype(t)
#     dims = SectorDict{I, Int}()
#
#     # compute LQ factorization for each block
#     V = domain(t)
#     if !isempty(blocksectors(V))
#         generator = Base.Iterators.map(blocksectors(V)) do c
#             Nc = MatrixAlgebra.rightnull!(block(t, c), alg, atol)
#             dims[c] = size(Nc, 1)
#             return c => Nc
#         end
#         Ndata = SectorDict(generator)
#     end
#
#     # construct new space
#     S = spacetype(t)
#     W = S(dims)
#
#     # construct output tensor
#     T = float(scalartype(t))
#     N = similar(t, T, W ← V)
#     if !isempty(blocksectors(V))
#         for (c, Nc) in Ndata
#             copy!(block(N, c), Nc)
#         end
#     end
#     return N
# end
# TK.rightnull!(t::SparseBlockTensorMap; kwargs...) = rightnull!(BlockTensorMap(t); kwargs...)
#
# function TK.tsvd!(t::AbstractBlockTensorMap; trunc = TK.NoTruncation(), p::Real = 2, alg = SDD())
#     return TK._tsvd!(t, alg, trunc, p)
# end
# function TK.tsvd!(t::SparseBlockTensorMap; kwargs...)
#     return tsvd!(BlockTensorMap(t); kwargs...)
# end
#
# function TK._tsvd!(
#         t::BlockTensorMap, alg::Union{SVD, SDD}, trunc::TruncationScheme, p::Real = 2
#     )
#     # early return
#     if isempty(blocksectors(t))
#         truncerr = zero(real(scalartype(t)))
#         return TK._empty_svdtensors(t)..., truncerr
#     end
#
#     # compute SVD factorization for each block
#     S = spacetype(t)
#     SVDdata, dims = TK._compute_svddata!(t, alg)
#     Σdata = SectorDict(c => Σ for (c, (U, Σ, V)) in SVDdata)
#     truncdim = TK._compute_truncdim(Σdata, trunc, p)
#     truncerr = TK._compute_truncerr(Σdata, truncdim, p)
#
#     # construct output tensors
#     U, Σ, V⁺ = TK._create_svdtensors(t, SVDdata, truncdim)
#     return U, Σ, V⁺, truncerr
# end
#
# function TK._compute_svddata!(t::AbstractBlockTensorMap, alg::Union{SVD, SDD})
#     InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
#     I = sectortype(t)
#     dims = SectorDict{I, Int}()
#     generator = Base.Iterators.map(TK.blocks(t)) do (c, b)
#         U, Σ, V = TK.MatrixAlgebra.svd!(b, alg)
#         dims[c] = length(Σ)
#         return c => (U, Σ, V)
#     end
#     SVDdata = SectorDict(generator)
#     return SVDdata, dims
# end
# function TK._create_svdtensors(t::AbstractBlockTensorMap, SVDdata, dims)
#     S = spacetype(t)
#     W = S(dims)
#     T = float(scalartype(t))
#     U = similar(t, T, codomain(t) ← W)
#     Σ = similar(t, real(T), W ← W)
#     V⁺ = similar(t, T, W ← domain(t))
#     for (c, (Uc, Σc, V⁺c)) in SVDdata
#         r = Base.OneTo(dims[c])
#         block(U, c) .= view(Uc, :, r)
#         block(Σ, c) .= Diagonal(view(Σc, r))
#         block(V⁺, c) .= view(V⁺c, r, :)
#     end
#     return U, Σ, V⁺
# end
#
# function TK._empty_svdtensors(t::AbstractBlockTensorMap)
#     T = scalartype(t)
#     S = spacetype(t)
#     I = sectortype(t)
#     dims = SectorDict{I, Int}()
#     W = S(dims)
#
#     U = similar(t, codomain(t) ← W)
#     Σ = similar(t, real(T), W ← W)
#     V⁺ = similar(t, W ← domain(t))
#     return U, Σ, V⁺
# end
