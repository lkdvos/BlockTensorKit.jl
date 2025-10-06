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
        t::AbstractBlockTensorMap{E, S, N₁, N₂}, f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}
    ) where {E, S, I, N₁, N₂}
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
@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractBlockArray, f₁::FusionTree, f₂::FusionTree
    )
    for I in eachindex(t)
        b = v[Block(I.I)]
        if !isempty(b)
            getindex!(t, I)[f₁, f₂] = v[Block(I.I)]
        end
    end
    return t
end
@inline function Base.setindex!(
        t::AbstractBlockTensorMap, v::AbstractArray, f₁::FusionTree, f₂::FusionTree
    )
    spaces = (codomain(t)..., domain(t)...)
    uncoupleds = (f₁.uncoupled..., f₂.uncoupled...)
    bsz = map(spaces, uncoupleds) do V, uncoupled
        return dim.(V, Ref(uncoupled))
    end
    v′ = BlockedArray(v, bsz...)
    return setindex!(t, v′, f₁, f₂)
end

function TensorKit.block(t::AbstractBlockTensorMap, c::Sector)
    sectortype(t) == typeof(c) || throw(SectorMismatch())

    rows = prod(TT.getindices(size(t), codomainind(t)))
    cols = prod(TT.getindices(size(t), domainind(t)))

    if rows == 0 || cols == 0
        allblocks = Matrix{TK.blocktype(eltype(t))}(undef, rows, cols)

        rowaxes = Int[]
        if rows != 0
            W′ = codomain(t) ← zero(spacetype(t))
            for V in eachspace(W′)
                push!(rowaxes, blockdim(codomain(V), c))
            end
        end

        colaxes = Int[]
        if cols != 0
            W′ = zero(spacetype(t)) ← domain(t)
            for V in eachspace(W′)
                push!(colaxes, blockdim(domain(V), c))
            end
        end

        return mortar(allblocks, rowaxes, colaxes)
    end

    allblocks = map(Base.Fix2(block, c), parent(t))
    return mortar(reshape(allblocks, rows, cols))
end

# TODO: this might get fixed once new tensormap is implemented
TensorKit.blocksectors(t::AbstractBlockTensorMap) = blocksectors(space(t))
TensorKit.hasblock(t::AbstractBlockTensorMap, c::Sector) = c in blocksectors(t)

TensorKit.blocks(t::AbstractBlockTensorMap) = TK.BlockIterator(t, blocksectors(t))
Base.@assume_effects :foldable function TensorKit.blocktype(::Type{TT}) where {TT <: AbstractBlockTensorMap}
    T = scalartype(TT)
    B = TK.blocktype(eltype(TT))
    (B <: AbstractMatrix{T}) || (B = AbstractMatrix{T}) # safeguard against type-instability
    BS = NTuple{2, BlockedOneTo{Int, Vector{Int}}}
    return BlockMatrix{T, Matrix{B}, BS}
end

function Base.iterate(iter::TK.BlockIterator{<:AbstractBlockTensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    c, newstate = next
    return c => block(iter.t, c), newstate
end
Base.getindex(iter::TK.BlockIterator{<:AbstractBlockTensorMap}, c::Sector) = block(iter.t, c)

function TensorKit.storagetype(::Type{TT}) where {TT <: AbstractBlockTensorMap}
    return if isconcretetype(eltype(TT))
        storagetype(eltype(TT))
    else
        Vector{scalartype(TT)}
    end
end
