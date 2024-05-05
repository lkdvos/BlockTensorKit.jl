# TensorKit Interface
# -------------------

TK.spacetype(::Union{T,Type{<:T}}) where {E,S,T<:BlockTensorMap{E,S}} = S
function TK.sectortype(::Union{T,Type{<:T}}) where {E,S,T<:BlockTensorMap{E,S}}
    return sectortype(S)
end
TK.storagetype(::Union{B,Type{B}}) where {E,S,N₁,N₂,N,B<:BlockTensorMap{E,S,N₁,N₂,N}} = AbstractTensorMap{E,S,N₁,N₂}
TK.storagetype(::Type{Union{A,B}}) where {A,B} = Union{storagetype(A),storagetype(B)}
function TK.similarstoragetype(TT::Type{<:BlockTensorMap}, ::Type{T}) where {T}
    return Core.Compiler.return_type(similar, Tuple{storagetype(TT),Type{T}})
end
TK.similarstoragetype(t::BlockTensorMap, T) = TK.similarstoragetype(typeof(t), T)

TK.numout(::Union{T,Type{T}}) where {E,S,N₁,T<:BlockTensorMap{E,S,N₁}} = N₁
TK.numin(::Union{T,Type{T}}) where {E,S,N₁,N₂,T<:BlockTensorMap{E,S,N₁,N₂}} = N₂
TK.numind(::Union{T,Type{T}}) where {E,S,N₁,N₂,T<:BlockTensorMap{E,S,N₁,N₂}} = N₁ + N₂

TK.codomain(t::BlockTensorMap) = t.codom
TK.domain(t::BlockTensorMap) = t.dom
TK.space(t::BlockTensorMap) = codomain(t) ← domain(t)
TK.space(t::BlockTensorMap, i) = space(t)[i]
TK.dim(t::BlockTensorMap) = dim(space(t))

function TK.blocksectors(t::BlockTensorMap)
    if eltype(t) isa TrivialTensorMap
        return OneOrNoneIterator(TK.dim(t) == 0, Trivial())
    else
        return blocksectors(codomain(t) ← domain(t))
    end
end

function TK.codomainind(::Union{T,Type{T}}) where {E,S,N₁,T<:BlockTensorMap{E,S,N₁}}
    return ntuple(n -> n, N₁)
end
function TK.domainind(::Union{T,Type{T}}) where {E,S,N₁,N₂,T<:BlockTensorMap{E,S,N₁,N₂}}
    return ntuple(n -> N₁ + n, N₂)
end
function TK.allind(::Union{T,Type{T}}) where {E,S,N₁,N₂,T<:BlockTensorMap{E,S,N₁,N₂}}
    return ntuple(n -> n, N₁ + N₂)
end

function TK.adjointtensorindex(::BlockTensorMap{<:Number,<:IndexSpace,N₁,N₂}, i) where {N₁,N₂}
    return ifelse(i <= N₁, N₂ + i, i - N₁)
end

function TK.adjointtensorindices(t::BlockTensorMap, indices::IndexTuple)
    return map(i -> TK.adjointtensorindex(t, i), indices)
end

function TK.adjointtensorindices(t::BlockTensorMap, p::Index2Tuple)
    return TK.adjointtensorindices(t, p[1]), TK.adjointtensorindices(t, p[2])
end

TK.blocks(t::BlockTensorMap) = ((c, block(t, c)) for c in blocksectors(t))

function TK.block(t::BlockTensorMap, c::Sector)
    sectortype(t) == typeof(c) || throw(SectorMismatch())

    rows = prod(getindices(size(t), codomainind(t)))
    cols = prod(getindices(size(t), domainind(t)))

    if rows == 0 || cols == 0
        error("to be added")
    end

    rowdims = subblockdims(codomain(t), c)
    coldims = subblockdims(domain(t), c)

    b = fill!(BlockArray{scalartype(t)}(undef, rowdims, coldims), zero(scalartype(t)))
    lin_inds = LinearIndices(parent(t))
    new_cart_inds = CartesianIndices((rows, cols))
    for (i, v) in nonzero_pairs(t)
        b[Block(new_cart_inds[lin_inds[i]].I)] = TK.block(v, c)
    end

    return b
end
