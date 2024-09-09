# TensorKit Interface
# -------------------

# TK.space(t::BlockTensorMap) = codomain(t) ← domain(t)
# TK.space(t::BlockTensorMap, i) = space(t)[i]
TK.dim(t::BlockTensorMap) = dim(space(t))

function TK.blocksectors(t::BlockTensorMap)
    if eltype(t) isa TrivialTensorMap
        return OneOrNoneIterator(TK.dim(t) == 0, Trivial())
    else
        return blocksectors(codomain(t) ← domain(t))
    end
end

# function TK.codomainind(::Union{T,Type{T}}) where {E,S,N₁,T<:BlockTensorMap{E,S,N₁}}
#     return ntuple(n -> n, N₁)
# end
# function TK.domainind(::Union{T,Type{T}}) where {E,S,N₁,N₂,T<:BlockTensorMap{E,S,N₁,N₂}}
#     return ntuple(n -> N₁ + n, N₂)
# end
# function TK.allind(::Union{T,Type{T}}) where {E,S,N₁,N₂,T<:BlockTensorMap{E,S,N₁,N₂}}
#     return ntuple(n -> n, N₁ + N₂)
# end

# function TK.adjointtensorindex(
#     ::BlockTensorMap{<:Number,<:IndexSpace,N₁,N₂}, i
# ) where {N₁,N₂}
#     return ifelse(i <= N₁, N₂ + i, i - N₁)
# end

# function TK.adjointtensorindices(t::BlockTensorMap, indices::IndexTuple)
#     return map(i -> TK.adjointtensorindex(t, i), indices)
# end
#
# function TK.adjointtensorindices(t::BlockTensorMap, p::Index2Tuple)
#     return TK.adjointtensorindices(t, p[1]), TK.adjointtensorindices(t, p[2])
# end

# TK.blocks(t::BlockTensorMap) = ((c, block(t, c)) for c in blocksectors(t))

# Note: this data is not in the same order as you would expect for a regular tensormap!
# function TK.block(t::BlockTensorMap, c::Sector)
#     sectortype(t) == typeof(c) || throw(SectorMismatch())
#
#     rows = prod(getindices(size(t), codomainind(t)))
#     cols = prod(getindices(size(t), domainind(t)))
#     @assert rows != 0 && cols != 0 "to be added"
#
#     allblocks = map(Base.Fix2(block, c), t.data)
#
#     return mortar(reshape(allblocks, rows, cols))
# end
#
# @inline function Base.getindex(
#     t::BlockTensorMap{E,S,N₁,N₂}, f₁::FusionTree{I,N₁}, f₂::FusionTree{I,N₂}
# ) where {E,S,I,N₁,N₂}
#     sectortype(S) === I || throw(SectorMismatch())
#     allblocks = map(x -> x[f₁, f₂], t.data)
#     return mortar(allblocks)
# end
#
# @inline function Base.getindex(t::BlockTensorMap, ::Nothing, ::Nothing)
#     return mortar(map(x -> x[nothing, nothing], t.data))
# end

# function Base.convert(::Type{TensorMap}, t::BlockTensorMap)
#     cod = ProductSpace{spacetype(t)}(join.(codomain(t).spaces))
#     dom = ProductSpace{spacetype(t)}(join.(domain(t).spaces))
#     tdst = TensorMap{scalartype(t)}(undef, cod ← dom)
#
#     for (f₁, f₂) in fusiontrees(tdst)
#         tdst[f₁, f₂] .= t[f₁, f₂]
#     end
#     return tdst
# end

# function TK.fusiontrees(t::BlockTensorMap)
#     sectortype(t) === Trivial && return ((nothing, nothing),)
#     blocksectoriterator = blocksectors(space(t))
#     rowr, _ = TK._buildblockstructure(codomain(t), blocksectoriterator)
#     colr, _ = TK._buildblockstructure(domain(t), blocksectoriterator)
#     return TK.TensorKeyIterator(rowr, colr)
# end

# function similarblocktype(
#     ::Type{TT}, ::Type{T}
# ) where {TT<:BlockTensorMap,T<:AbstractTensorMap}
#     return Core.Compiler.return_type(similar, Tuple{blocktype(TT),Type{T}})
# end

# function similarstoragetype(TT::Type{<:BlockTensorMap}, ::Type{T}) where {T<:Number}
#     TT′ = tensormaptype(spacetype(TT), numout(TT), numin(TT), T)
#     return Core.Compiler.return_type(similar,
#                                      Tuple{storagetype(TT),Type{TT′},
#                                            NTuple{numind(TT),Int}})
# end
