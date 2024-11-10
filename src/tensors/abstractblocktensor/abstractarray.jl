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

Base.CartesianIndices(t::AbstractBlockTensorMap) = CartesianIndices(parent(t))
Base.eachindex(t::AbstractBlockTensorMap) = eachindex(parent(t))

Base.keys(l::Base.IndexStyle, t::AbstractBlockTensorMap) = keys(l, parent(t))
Base.haskey(t::AbstractBlockTensorMap, args...) = haskey(parent(t), args...)

Base.only(t::AbstractBlockTensorMap) = only(parent(t))
Base.isempty(t::AbstractBlockTensorMap) = isempty(parent(t))

# index checking
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

# scalar indexing is dispatched through:
@inline Base.getindex(t::AbstractBlockTensorMap, I::Vararg{Int,N}) where {N} =
    getindex(parent(t), I...)
@inline Base.getindex(t::AbstractBlockTensorMap, I::CartesianIndex{N}) where {N} =
    getindex(parent(t), I)

# slicing getindex needs to correctly allocate output blocktensor:
const SliceIndex = Union{Strided.SliceIndex,AbstractVector{<:Union{Integer,Bool}}}

Base.@propagate_inbounds function Base.getindex(
    t::AbstractBlockTensorMap, indices::Vararg{SliceIndex}
)
    V = space(eachspace(t)[indices...])
    tdst = similar(t, V)
    copyto!(parent(tdst), view(parent(t), indices...))
    return tdst
end
# disambiguate:
Base.@propagate_inbounds function Base.getindex(
    t::AbstractBlockTensorMap, indices::Vararg{Strided.SliceIndex}
)
    V = space(eachspace(t)[indices...])
    tdst = similar(t, V)
    copyto!(parent(tdst), view(parent(t), indices...))
    return tdst
end

# TODO: check if this fallback is fair
@inline Base.setindex!(t::AbstractBlockTensorMap, args...) = (
    setindex!(parent(t), args...); t
)

# setindex verifies structure is correct
@inline function Base.setindex!(
    t::AbstractBlockTensorMap, v::AbstractTensorMap, indices::Vararg{SliceIndex}
)
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end
    @inbounds parent(t)[indices...] = v
    return t
end
# setindex with blocktensor needs to correctly slice-assign
@inline function Base.setindex!(
    t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, indices::Vararg{SliceIndex}
)
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end

    copyto!(view(parent(t), indices...), parent(v))
    return t
end

# disambiguate
@inline function Base.setindex!(
    t::AbstractBlockTensorMap, v::AbstractTensorMap, indices::Vararg{Strided.SliceIndex}
)
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end
    @inbounds parent(t)[indices...] = v
    return t
end
# disambiguate
@inline function Base.setindex!(
    t::AbstractBlockTensorMap,
    v::AbstractBlockTensorMap,
    indices::Vararg{Strided.SliceIndex},
)
    @boundscheck begin
        checkbounds(t, indices...)
        checkspaces(t, v, indices...)
    end

    copyto!(view(parent(t), indices...), parent(v))
    return t
end

@inline function Base.get(t::AbstractBlockTensorMap, key, default)
    @boundscheck checkbounds(t, key)
    return get(parent(t), key, default)
end

Base.copy(t::AbstractBlockTensorMap) = copy!(similar(t), t)
function Base.copy!(tdst::AbstractBlockTensorMap, tsrc::AbstractBlockTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    @inbounds for (key, value) in nonzero_pairs(tsrc)
        tdst[key] = value
    end
    return tdst
end
function Base.copyto!(
    tdst::AbstractBlockTensorMap,
    Rdest::CartesianIndices,
    tsrc::AbstractBlockTensorMap,
    Rsrc::CartesianIndices,
)
    copyto!(parent(tdst), Rdest, parent(tsrc), Rsrc)
    return tdst
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
    t::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S}
) where {S,TorA}
    if TorA <: AbstractTensorMap
        return BlockTensorMap{TorA}(undef_blocks, P)
    elseif TorA <: Number
        T = TorA
        A = TensorKit.similarstoragetype(t, T)
    elseif TorA <: DenseVector
        A = TorA
        T = scalartype(A)
    else
        throw(ArgumentError("Type $TorA not supported for similar"))
    end
    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    TT = TensorMap{T,S,N₁,N₂,A}
    return BlockTensorMap{TT}(undef, P)
end

function Base.similar(::Type{T}, P::TensorMapSumSpace) where {T<:AbstractBlockTensorMap}
    return T(undef_blocks, P)
end