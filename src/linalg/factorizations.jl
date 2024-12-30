using TensorKit: QR, QRpos, QL, QLpos, SVD, SDD, Polar, LQ, LQpos, RQ, RQpos

function TK.leftorth!(
    t::AbstractBlockTensorMap;
    alg::Union{QR,QRpos,QL,QLpos,SVD,SDD,Polar}=QRpos(),
    atol::Real=zero(float(real(scalartype(t)))),
    rtol::Real=if (alg ∉ (SVD(), SDD()))
        zero(float(real(scalartype(t))))
    else
        eps(real(float(one(scalartype(t))))) * iszero(atol)
    end,
)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = TK.SectorDict{I,Int}()

    # compute QR factorization for each block
    if !isempty(TK.blocks(t))
        generator = Base.Iterators.map(TK.blocks(t)) do (c, b)
            Qc, Rc = TK.MatrixAlgebra.leftorth!(b, alg, atol)
            dims[c] = size(Qc, 2)
            return c => (Qc, Rc)
        end
        QRdata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    V = S(dims)
    if alg isa Polar
        @assert V ≅ domain(t)
        W = domain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    else
        W = ProductSpace(V)
    end

    # construct output tensors
    T = float(scalartype(t))
    Q = similar(t, T, codomain(t) ← W)
    R = similar(t, T, W ← domain(t))
    if !isempty(blocksectors(domain(t)))
        for (c, (Qc, Rc)) in QRdata
            block(Q, c) .= Qc
            block(R, c) .= Rc
        end
    end
    return Q, R
end
function TK.leftorth!(t::SparseBlockTensorMap; kwargs...)
    return leftorth!(BlockTensorMap(t); kwargs...)
end

function TK.leftnull!(
    t::BlockTensorMap;
    alg::Union{QR,QRpos,SVD,SDD}=QRpos(),
    atol::Real=zero(float(real(scalartype(t)))),
    rtol::Real=if (alg ∉ (SVD(), SDD()))
        zero(float(real(scalartype(t))))
    else
        eps(real(float(one(scalartype(t))))) * iszero(atol)
    end,
)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftnull!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute QR factorization for each block
    V = codomain(t)
    if !isempty(blocksectors(V))
        generator = Base.Iterators.map(blocksectors(V)) do c
            Nc = MatrixAlgebra.leftnull!(block(t, c), alg, atol)
            dims[c] = size(Nc, 2)
            return c => Nc
        end
        Ndata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    W = S(dims)

    # construct output tensor
    T = float(scalartype(t))
    N = similar(t, T, V ← W)
    if !isempty(blocksectors(V))
        for (c, Nc) in Ndata
            copy!(block(N, c), Nc)
        end
    end
    return N
end
TK.leftnull!(t::SparseBlockTensorMap; kwargs...) = leftnull!(BlockTensorMap(t); kwargs...)

function TK.rightorth!(
    t::AbstractBlockTensorMap;
    alg::Union{LQ,LQpos,RQ,RQpos,SVD,SDD,Polar}=LQpos(),
    atol::Real=zero(float(real(scalartype(t)))),
    rtol::Real=if (alg ∉ (SVD(), SDD()))
        zero(float(real(scalartype(t))))
    else
        eps(real(float(one(scalartype(t))))) * iszero(atol)
    end,
)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = TK.SectorDict{I,Int}()

    # compute LQ factorization for each block
    if !isempty(TK.blocks(t))
        generator = Base.Iterators.map(TK.blocks(t)) do (c, b)
            Lc, Qc = TK.MatrixAlgebra.rightorth!(b, alg, atol)
            dims[c] = size(Qc, 1)
            return c => (Lc, Qc)
        end
        LQdata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    V = S(dims)
    if alg isa Polar
        @assert V ≅ codomain(t)
        W = codomain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    else
        W = ProductSpace(V)
    end

    # construct output tensors
    T = float(scalartype(t))
    L = similar(t, T, codomain(t) ← W)
    Q = similar(t, T, W ← domain(t))
    if !isempty(blocksectors(codomain(t)))
        for (c, (Lc, Qc)) in LQdata
            block(L, c) .= Lc
            block(Q, c) .= Qc
        end
    end
    return L, Q
end
function TK.rightorth!(t::SparseBlockTensorMap; kwargs...)
    return rightorth!(BlockTensorMap(t); kwargs...)
end

function TK.rightnull!(
    t::BlockTensorMap;
    alg::Union{LQ,LQpos,SVD,SDD}=LQpos(),
    atol::Real=zero(float(real(scalartype(t)))),
    rtol::Real=if (alg ∉ (SVD(), SDD()))
        zero(float(real(scalartype(t))))
    else
        eps(real(float(one(scalartype(t))))) * iszero(atol)
    end,
)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightnull!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute LQ factorization for each block
    V = domain(t)
    if !isempty(blocksectors(V))
        generator = Base.Iterators.map(blocksectors(V)) do c
            Nc = MatrixAlgebra.rightnull!(block(t, c), alg, atol)
            dims[c] = size(Nc, 1)
            return c => Nc
        end
        Ndata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    W = S(dims)

    # construct output tensor
    T = float(scalartype(t))
    N = similar(t, T, W ← V)
    if !isempty(blocksectors(V))
        for (c, Nc) in Ndata
            copy!(block(N, c), Nc)
        end
    end
    return N
end
TK.rightnull!(t::SparseBlockTensorMap; kwargs...) = rightnull!(BlockTensorMap(t); kwargs...)

function TK.tsvd!(t::AbstractBlockTensorMap; trunc=TK.NoTruncation(), p::Real=2, alg=SDD())
    return TK._tsvd!(t, alg, trunc, p)
end
function TK.tsvd!(t::SparseBlockTensorMap; kwargs...)
    return tsvd!(BlockTensorMap(t); kwargs...)
end

function TK._tsvd!(
    t::BlockTensorMap, alg::Union{SVD,SDD}, trunc::TruncationScheme, p::Real=2
)
    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return TK._empty_svdtensors(t)..., truncerr
    end

    # compute SVD factorization for each block
    S = spacetype(t)
    SVDdata, dims = TK._compute_svddata!(t, alg)
    Σdata = SectorDict(c => Σ for (c, (U, Σ, V)) in SVDdata)
    truncdim = TK._compute_truncdim(Σdata, trunc, p)
    truncerr = TK._compute_truncerr(Σdata, truncdim, p)

    # construct output tensors
    U, Σ, V⁺ = TK._create_svdtensors(t, SVDdata, truncdim)
    return U, Σ, V⁺, truncerr
end

function TK._compute_svddata!(t::AbstractBlockTensorMap, alg::Union{SVD,SDD})
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(t)
    dims = SectorDict{I,Int}()
    generator = Base.Iterators.map(TK.blocks(t)) do (c, b)
        U, Σ, V = TK.MatrixAlgebra.svd!(b, alg)
        dims[c] = length(Σ)
        return c => (U, Σ, V)
    end
    SVDdata = SectorDict(generator)
    return SVDdata, dims
end
function TK._create_svdtensors(t::AbstractBlockTensorMap, SVDdata, dims)
    S = spacetype(t)
    W = S(dims)
    T = float(scalartype(t))
    U = similar(t, T, codomain(t) ← W)
    Σ = similar(t, real(T), W ← W)
    V⁺ = similar(t, T, W ← domain(t))
    for (c, (Uc, Σc, V⁺c)) in SVDdata
        r = Base.OneTo(dims[c])
        block(U, c) .= view(Uc, :, r)
        block(Σ, c) .= Diagonal(view(Σc, r))
        block(V⁺, c) .= view(V⁺c, r, :)
    end
    return U, Σ, V⁺
end

function TK._empty_svdtensors(t::AbstractBlockTensorMap)
    T = scalartype(t)
    S = spacetype(t)
    I = sectortype(t)
    dims = SectorDict{I,Int}()
    W = S(dims)

    U = similar(t, codomain(t) ← W)
    Σ = similar(t, real(T), W ← W)
    V⁺ = similar(t, W ← domain(t))
    return U, Σ, V⁺
end
