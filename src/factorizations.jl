using TensorKit: QR, QRpos, QL, QLpos, SVD, SDD, Polar, LQ, LQpos, RQ, RQpos

function TK.leftorth!(t::BlockTensorMap;
                      alg::Union{QR,QRpos,QL,QLpos,SVD,SDD,Polar}=QRpos(),
                      atol::Real=zero(float(real(scalartype(t)))),
                      rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                                 eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:leftorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    Qdata = TK.SectorDict{I,A}()
    Rdata = TK.SectorDict{I,A}()
    dims = TK.SectorDict{I,Int}()
    for c in blocksectors(domain(t))
        isempty(block(t, c)) && continue
        Q, R = TK.MatrixAlgebra.leftorth!(block(t, c), alg, atol)
        Qdata[c] = Q
        Rdata[c] = R
        dims[c] = size(Q, 2)
    end
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

    Q = BlockTensorMap{scalartype(t)}(undef, codomain(t) ← W)
    R = BlockTensorMap{scalartype(t)}(undef, W ← domain(t))

    for c in blocksectors(domain(t))
        block(Q, c) .= Qdata[c]
        block(R, c) .= Rdata[c]
    end

    return Q, R
end

function TK.rightorth!(t::BlockTensorMap;
                       alg::Union{LQ,LQpos,RQ,RQpos,SVD,SDD,Polar}=LQpos(),
                       atol::Real=zero(float(real(scalartype(t)))),
                       rtol::Real=(alg ∉ (SVD(), SDD())) ?
                                  zero(float(real(scalartype(t)))) :
                                  eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:rightorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    Ldata = TK.SectorDict{I,A}()
    Qdata = TK.SectorDict{I,A}()
    dims = TK.SectorDict{I,Int}()
    for c in blocksectors(codomain(t))
        isempty(block(t, c)) && continue
        L, Q = TK.MatrixAlgebra.rightorth!(block(t, c), alg, atol)
        Ldata[c] = L
        Qdata[c] = Q
        dims[c] = size(Q, 1)
    end
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

    L = BlockTensorMap{scalartype(t)}(undef, codomain(t) ← W)
    Q = BlockTensorMap{scalartype(t)}(undef, W ← domain(t))
    for c in blocksectors(codomain(t))
        block(L, c) .= Ldata[c]
        block(Q, c) .= Qdata[c]
    end
    return L, Q
end

function TK.tsvd!(t::BlockTensorMap; trunc=NoTruncation(), p::Real=2, alg=SDD())
    return TK._tsvd!(t, alg, trunc, p)
end

function TK._compute_svddata!(t::BlockTensorMap, alg::Union{SVD,SDD})
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(t)
    A = storagetype(t)
    Udata = TK.SectorDict{I,A}()
    Vdata = TK.SectorDict{I,A}()
    dims = TK.SectorDict{I,Int}()
    local Σdata
    for (c, b) in TK.blocks(t)
        U, Σ, V = MatrixAlgebra.svd!(b, alg)
        Udata[c] = U
        Vdata[c] = V
        if @isdefined Σdata # cannot easily infer the type of Σ, so use this construction
            Σdata[c] = Σ
        else
            Σdata = TK.SectorDict(c => Σ)
        end
        dims[c] = length(Σ)
    end
    return Udata, Σdata, Vdata, dims
end

function TK._empty_svdtensors(t::BlockTensorMap)
    S = spacetype(t)
    A = storagetype(t)
    Ar = similarstoragetype(t, real(scalartype(t)))
    W = S(dims)
    TU = tensormaptype(sumspacetype(S), numout(t), 1, A)
    U = TU(undef, codomain(t) ← W)
    TΣ = tensormaptype(sumspacetype(S), 1, 1, Ar)
    Σ = TΣ(undef, W ← W)
    TV = tensormaptype(sumspacetype(S), 1, numin(t), A)
    V = TV(undef, W ← domain(t))

    return U, Σ, V
end

function TK._create_svdtensors(t::BlockTensorMap, Udata, Σdata, Vdata, W)
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    Ar = TK.similarstoragetype(t, real(scalartype(t)))
    Σmdata = SectorDict{I,Ar}() # this will contain the singular values as matrix
    for (c, Σ) in Σdata
        Σmdata[c] = copyto!(similar(Σ, length(Σ), length(Σ)), Diagonal(Σ))
    end

    TU = tensormaptype(sumspacetype(S), numout(t), 1, A)
    U = TU(undef, codomain(t) ← W)
    for (key, value) in Udata
        block(U, key) .= value
    end

    TΣ = tensormaptype(sumspacetype(S), 1, 1, Ar)
    Σ = TΣ(undef, W ← W)
    for (key, value) in Σdata
        block(Σ, key) .= value
    end

    TV = tensormaptype(sumspacetype(S), 1, numin(t), A)
    V = TV(undef, W ← domain(t))
    for (key, value) in Vdata
        block(V, key) .= value
    end

    return U, Σ, V
end
