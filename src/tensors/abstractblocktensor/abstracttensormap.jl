
# AbstractTensorMap Interface
# ---------------------------
# TODO: do we really want this:
# note: this goes along with the specializations of Base.similar above...
function TensorKit.tensormaptype(
    ::Type{SumSpace{S}}, N₁::Int, N₂::Int, TorA::Type
) where {S}
    return blocktensormaptype(S, N₁, N₂, TorA)
end

eachspace(t::AbstractBlockTensorMap) = SumSpaceIndices(space(t))

# TODO: delete this method
@inline function Base.getindex(t::AbstractBlockTensorMap, ::Nothing, ::Nothing)
    sectortype(t) === Trivial || throw(SectorMismatch())
    return mortar(map(x -> x[nothing, nothing], parent(t)))
end
@inline function Base.getindex(
    t::AbstractBlockTensorMap{E,S,N₁,N₂}, f₁::FusionTree{I,N₁}, f₂::FusionTree{I,N₂}
) where {E,S,I,N₁,N₂}
    sectortype(S) === I || throw(SectorMismatch())
    subblocks = map(eachspace(t), parent(t)) do V, x
        sz = (dims(codomain(V), f₁.uncoupled)..., dims(domain(V), f₂.uncoupled)...)
        if prod(sz) == 0
            data = storagetype(t)(undef, 0)
            return sreshape(StridedView(data), sz)
        else
            return x[f₁, f₂]
        end
    end
    return mortar(subblocks)
end

function TensorKit.block(t::AbstractBlockTensorMap, c::Sector)
    sectortype(t) == typeof(c) || throw(SectorMismatch())

    rows = prod(TT.getindices(size(t), codomainind(t)))
    cols = prod(TT.getindices(size(t), domainind(t)))
    @assert rows != 0 && cols != 0 "to be added"

    allblocks = map(Base.Fix2(block, c), parent(t))
    return mortar(reshape(allblocks, rows, cols))
end

# TODO: this might get fixed once new tensormap is implemented
TensorKit.blocks(t::AbstractBlockTensorMap) = ((c, block(t, c)) for c in blocksectors(t))
TensorKit.blocksectors(t::AbstractBlockTensorMap) = blocksectors(space(t))
TensorKit.hasblock(t::AbstractBlockTensorMap, c::Sector) = c in blocksectors(t)

function TensorKit.storagetype(::Type{TT}) where {TT<:AbstractBlockTensorMap}
    return if isconcretetype(eltype(TT))
        storagetype(eltype(TT))
    else
        Vector{scalartype(TT)}
    end
end
