"""
    AbstractBlockTensorMap{E,S,N₁,N₂}

Abstract supertype for tensor maps that have additional block structure, i.e. they act on vector spaces
that have a direct sum structure. These behave like `AbstractTensorMap` but have additional methods to
facilitate indexing and manipulation of the block structure.
"""
abstract type AbstractBlockTensorMap{E,S,N₁,N₂} <: AbstractTensorMap{E,S,N₁,N₂} end

# AbstractArray Interface
# -----------------------
# mostly pass everything through to the parent array, but with additional features for slicing
Base.eltype(t::AbstractBlockTensorMap) = eltype(parent(t))

Base.ndims(t::AbstractBlockTensorMap) = ndims(parent(t))
Base.size(t::AbstractBlockTensorMap, args...) = size(parent(t), args...)
Base.length(t::AbstractBlockTensorMap) = length(parent(t))
Base.axes(t::AbstractBlockTensorMap, args...) = axes(parent(t), args...)

Base.first(t::AbstractBlockTensorMap) = first(parent(t))
Base.last(t::AbstractBlockTensorMap) = last(parent(t))
Base.lastindex(t::AbstractBlockTensorMap, args...) = lastindex(parent(t), args...)
Base.firstindex(t::AbstractBlockTensorMap, args...) = firstindex(parent(t), args...)

Base.keys(l::IndexStyle, t::AbstractBlockTensorMap) = keys(l, parent(t))

Base.only(t::AbstractBlockTensorMap) = only(parent(t))
Base.isempty(t::AbstractBlockTensorMap) = isempty(parent(t))

# scalar indexing is dispatched through:
@inline Base.getindex(t::AbstractBlockTensorMap, I::Vararg{Int,N}) where {N} =
    getindex(parent(t), I...)
@inline Base.getindex(t::AbstractBlockTensorMap, I::CartesianIndex{N}) where {N} =
    getindex(parent(t), I)

# slicing getindex needs to correctly allocate output blocktensor:
Base.@propagate_inbounds function Base.getindex(t::AbstractBlockTensorMap, I...)
    V = space(eachspace(t)[I...])
    tdst = similar(t, V)
    copyto!(parent(tdst), view(parent(t), I...))
    return tdst
end

# TODO: check if this fallback is fair
@inline Base.setindex!(t::AbstractBlockTensorMap, args...) = (
    setindex!(parent(t), args...); t
)

# setindex verifies structure is correct
@inline function Base.setindex!(t::AbstractBlockTensorMap, v::AbstractTensorMap, I...)
    @boundscheck begin
        checkbounds(t, I...)
        checkspaces(t, v, I...)
    end
    @inbounds parent(t)[I...] = v
    return t
end
# setindex with blocktensor needs to correctly slice-assign
@inline function Base.setindex!(t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, I...)
    @boundscheck begin
        checkbounds(t, I...)
        checkspaces(t, v, I...)
    end

    copyto!(view(parent(t), I...), parent(v))
    return t
end

# generic implementation for AbstractTensorMap with Sumspace -> returns `BlockTensorMap`
# function Base.similar(
#     ::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S}
# ) where {TorA<:TensorKit.MatOrNumber,S}
#     N₁ = length(codomain(P))
#     N₂ = length(domain(P))
#     TT = blocktensormaptype(S, N₁, N₂, TorA)
#     return TT(undef, codomain(P), domain(P))
# end
# disambiguate
# function Base.similar(
#     t::TensorKit.AdjointTensorMap, T::Type{TorA}, P::TensorMapSumSpace{S}
# ) where {TorA<:TensorKit.MatOrNumber,S}
#     @invoke Base.similar(t::TensorKit.AdjointTensorMap, T::Type{TorA}, P::TensorMapSpace)
# end

# make sure tensormap specializations are not used for sumspaces:
function Base.similar(
    t::TensorMap, T::Type{TorA}, P::TensorMapSumSpace{S}
) where {TorA<:TensorKit.MatOrNumber,S}
    @invoke Base.similar(t::AbstractTensorMap, T::Type{TorA}, P)
end

# AbstractTensorMap Interface
# ---------------------------
# TODO: do we really want this:
# note: this goes along with the specializations of Base.similar above...
function TensorKit.tensormaptype(
    ::Type{SumSpace{S}}, N₁::Int, N₂::Int, ::Type{TorA}
) where {S,TorA<:TensorKit.MatOrNumber}
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
    return mortar(map(x -> x[f₁, f₂], parent(t)))
end

function TensorKit.block(t::AbstractBlockTensorMap, c::Sector)
    sectortype(t) == typeof(c) || throw(SectorMismatch())

    rows = prod(getindices(size(t), codomainind(t)))
    cols = prod(getindices(size(t), domainind(t)))
    @assert rows != 0 && cols != 0 "to be added"

    allblocks = map(Base.Fix2(block, c), parent(t))
    return mortar(reshape(allblocks, rows, cols))
end

# TODO: this might get fixed once new tensormap is implemented
TensorKit.blocks(t::AbstractBlockTensorMap) = ((c, block(t, c)) for c in blocksectors(t))
TensorKit.blocksectors(t) = blocksectors(space(t))
TensorKit.hasblock(t::AbstractBlockTensorMap, c::Sector) = c in blocksectors(t)

function TensorKit.storagetype(::Type{TT}) where {TT<:AbstractBlockTensorMap}
    return if isconcretetype(eltype(TT))
        storagetype(eltype(TT))
    else
        Matrix{scalartype(TT)}
    end
end

function TensorKit.fusiontrees(t::AbstractBlockTensorMap)
    sectortype(t) === Trivial && return ((nothing, nothing),)
    blocksectoriterator = blocksectors(space(t))
    rowr, _ = TK._buildblockstructure(codomain(t), blocksectoriterator)
    colr, _ = TK._buildblockstructure(domain(t), blocksectoriterator)
    return TK.TensorKeyIterator(rowr, colr)
end

# getindex and setindex checking
# ------------------------------
Base.checkbounds(t::AbstractBlockTensorMap, I...) = checkbounds(parent(t), I...)
function Base.checkbounds(::Type{Bool}, t::AbstractBlockTensorMap, I...)
    return checkbounds(Bool, parent(t), I...)
end

# TODO: make this also have Bool as first argument
function checkspaces(t::AbstractBlockTensorMap, v::AbstractTensorMap, I...)
    space(v) == eachspace(t)[I...] || throw(
        SpaceMismatch(
            "inserting a tensor of space $(space(v)) at index $I into a tensor of space $(eachspace(t)[I...])",
        ),
    )
    return nothing
end
function checkspaces(t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, I...)
    V_slice = eachspace(t)[I...]
    if V_slice isa SumSpaceIndices
        space(v) == space(V_slice) || throw(
            SpaceMismatch(
                "inserting a tensor of space $(space(v)) at index $I into a tensor of space $(space(V_slice))",
            ),
        )
    else
        space(only(v)) == V_slice || throw(
            SpaceMismatch(
                "inserting a tensor of space $(space(only(v))) at index $I into a tensor of space $(V_slice)",
            ),
        )
    end
end
function checkspaces(t::AbstractBlockTensorMap)
    iter = SumSpaceIndices(space(t))
    for I in eachindex(iter)
        iter[I] == space(t[I]) || throw(
            SpaceMismatch(
                "index $I has space $(iter[I]) but tensor has space $(space(t[I]))"
            ),
        )
    end
    return nothing
end

# Conversion
# ----------
function Base.convert(::Type{T}, t::AbstractBlockTensorMap) where {T<:TensorMap}
    cod = ProductSpace{spacetype(t),numout(t)}(join.(codomain(t).spaces))
    dom = ProductSpace{spacetype(t),numin(t)}(join.(domain(t).spaces))

    tdst = similar(t, cod ← dom)
    for (f₁, f₂) in fusiontrees(tdst)
        tdst[f₁, f₂] .= t[f₁, f₂]
    end

    return convert(T, tdst)
end

# Sparsity
# --------

nonzero_pairs(t::AbstractBlockTensorMap) = nonzero_pairs(parent(t))
nonzero_keys(t::AbstractBlockTensorMap) = nonzero_keys(parent(t))
nonzero_values(t::AbstractBlockTensorMap) = nonzero_values(parent(t))
nonzero_length(t::AbstractBlockTensorMap) = nonzero_length(parent(t))

nonzero_values(A::AbstractArray) = values(A)
nonzero_keys(A::AbstractArray) = keys(A)
nonzero_pairs(A::AbstractArray) = pairs(A)
nonzero_length(A::AbstractArray) = length(A)
