# SparseBlockTensorMap parent array
# ---------------------------------
struct SparseTensorArray{S,N₁,N₂,T<:AbstractTensorMap{<:Any,S,N₁,N₂},N} <:
       AbstractArray{T,N}
    data::Dict{CartesianIndex{N},T}
    space::TensorMapSumSpace{S,N₁,N₂}
    function SparseTensorArray{S,N₁,N₂,T,N}(
        data::Dict{CartesianIndex{N},T}, space::TensorMapSumSpace{S,N₁,N₂}
    ) where {S,N₁,N₂,T,N}
        N₁ + N₂ == N || throw(
            TypeError(
                :SparseTensorArray,
                SparseTensorArray{S,N₁,N₂,T,N₁ + N₂},
                SparseTensorArray{S,N₁,N₂,T,N},
            ),
        )
        return new{S,N₁,N₂,T,N}(data, space)
    end
end

function SparseTensorArray{S,N₁,N₂,T,N}(
    ::UndefInitializer, space::TensorMapSumSpace{S,N₁,N₂}
) where {S,N₁,N₂,T<:AbstractTensorMap{<:Any,S,N₁,N₂},N}
    return SparseTensorArray{S,N₁,N₂,T,N}(Dict{CartesianIndex{N},T}(), space)
end

function SparseTensorArray(
    data::Dict{CartesianIndex{N},T}, space::TensorMapSumSpace{S,N₁,N₂}
) where {S,N₁,N₂,T,N}
    return SparseTensorArray{S,N₁,N₂,T,N}(data, space)
end

Base.pairs(A::SparseTensorArray) = pairs(A.data)
Base.keys(A::SparseTensorArray) = keys(A.data)
Base.values(A::SparseTensorArray) = values(A.data)

TensorKit.space(A::SparseTensorArray) = A.space

# AbstractArray interface
# -----------------------
Base.size(A::SparseTensorArray) = ntuple(i -> length(space(A)[i]), ndims(A))

@inline function Base.getindex(
    A::SparseTensorArray{S,N₁,N₂,T,N}, I::Vararg{Int,N}
) where {S,N₁,N₂,T,N}
    @boundscheck checkbounds(A, I...)
    return @inbounds get(A.data, CartesianIndex(I)) do
        return fill!(similar(T, eachspace(A)[I...]), zero(scalartype(T)))
    end
end
@inline function getindex!(
    A::SparseTensorArray{S,N₁,N₂,T,N}, I::CartesianIndex{N}
) where {S,N₁,N₂,T,N}
    @boundscheck checkbounds(A, I)
    return @inbounds get!(A.data, I) do
        return fill!(similar(T, eachspace(A)[I]), zero(scalartype(T)))
    end
end
@inline function getindex!(
    A::SparseTensorArray{S,N₁,N₂,T,N}, I::Vararg{Int,N}
) where {S,N₁,N₂,T,N}
    return getindex!(A, CartesianIndex(I))
end
@inline function Base.setindex!(
    A::SparseTensorArray{S,N₁,N₂,T,N}, v, I::Vararg{Int,N}
) where {S,N₁,N₂,T,N}
    @boundscheck begin
        checkbounds(A, I...)
        checkspaces(A, v, I...)
    end
    @inbounds A.data[CartesianIndex(I)] = v # implicit converter
    return A
end

function Base.delete!(A::SparseTensorArray, I::Vararg{Int,N}) where {N}
    return delete!(A.data, CartesianIndex(I))
end
Base.delete!(A::SparseTensorArray, I::CartesianIndex) = delete!(A.data, I)
Base.empty!(A::SparseTensorArray) = empty!(A.data)
function Base.haskey(A::SparseTensorArray, I::Vararg{Int,N}) where {N}
    return haskey(A.data, CartesianIndex(I))
end
Base.haskey(A::SparseTensorArray, I::CartesianIndex) = haskey(A.data, I)

function Base.similar(
    ::SparseTensorArray, ::Type{T}, spaces::TensorMapSumSpace{S,N₁,N₂}
) where {S,N₁,N₂,T<:AbstractTensorMap{<:Any,S,N₁,N₂}}
    N = N₁ + N₂
    return SparseTensorArray{S,N₁,N₂,T,N}(Dict{CartesianIndex{N},T}(), spaces)
end

Base.@propagate_inbounds function Base.copyto!(
    t::SparseTensorArray, v::SubArray{T,N,A}
) where {T,N,A<:SparseTensorArray}
    undropped_parentindices = map(Base.parentindices(v)) do I
        I isa Base.ScalarIndex ? (I:I) : I
    end

    for I in eachindex(IndexCartesian(), t)
        parentI = CartesianIndex(Base.reindex(undropped_parentindices, I.I))
        if haskey(parent(v), parentI)
            t[I] = parent(v)[parentI]
        else
            delete!(t, I)
        end
    end
    return t
end

Base.@propagate_inbounds function Base.copyto!(
    t::SubArray{T,N,A}, v::SparseTensorArray
) where {T,N,A<:SparseTensorArray}
    undropped_parentindices = map(Base.parentindices(t)) do I
        I isa Base.ScalarIndex ? (I:I) : I
    end

    for I in eachindex(IndexCartesian(), v)
        if haskey(v, I)
            t[I] = v[I]
        else
            parentI = CartesianIndex(Base.reindex(undropped_parentindices, I.I))
            delete!(parent(t), parentI)
        end
    end
    return t
end

Base.@propagate_inbounds function Base.copyto!(
    dest::SparseTensorArray,
    Rdest::CartesianIndices,
    src::SparseTensorArray,
    Rsrc::CartesianIndices,
)
    isempty(Rdest) && return dest
    if size(Rdest) != size(Rsrc)
        throw(
            ArgumentError(
                "source and destination must have same size (got $(size(Rsrc)) and $(size(Rdest)))",
            ),
        )
    end
    @boundscheck begin
        checkbounds(dest, first(Rdest))
        checkbounds(dest, last(Rdest))
        checkbounds(src, first(Rsrc))
        checkbounds(src, last(Rsrc))
    end
    CRdest = CartesianIndices(Rdest)
    CRsrc = CartesianIndices(Rsrc)
    ΔI = first(CRdest) - first(CRsrc)
    for I in CRsrc
        if Rsrc[I] in nonzero_keys(src)
            dest[Rdest[I + ΔI]] = src[Rsrc[I]]
        end
    end
    return dest
end

# non-scalar indexing
# -------------------
# specialisations to have non-scalar indexing behave as expected

_newindex(i::Int, range::Int) = i == range ? (1,) : nothing
function _newindex(i::Int, range::AbstractVector{Int})
    k = findfirst(==(i), range)
    return k === nothing ? nothing : (k,)
end
_newindices(::Tuple{}, ::Tuple{}) = ()
function _newindices(I::Tuple, indices::Tuple)
    i = _newindex(I[1], indices[1])
    Itail = _newindices(Base.tail(I), Base.tail(indices))
    (i === nothing || Itail === nothing) && return nothing
    return (i..., Itail...)
end

function Base._unsafe_getindex(
    ::IndexCartesian,
    t::SparseTensorArray{S,N₁,N₂,T,N},
    I::Vararg{Union{Real,AbstractArray},N},
) where {S,N₁,N₂,T,N}
    dest = similar(t, eltype(t), space(eachspace(t)[I...]))
    indices = Base.to_indices(t, I)
    for (k, v) in t.data
        newI = _newindices(k.I, indices)
        if newI !== nothing
            dest[newI...] = v
        end
    end
    return dest
end

# Space checking
# --------------
eachspace(A::SparseTensorArray) = SumSpaceIndices(A.space)
function checkspaces(A::SparseTensorArray, v::AbstractTensorMap, I...)
    return space(v) == eachspace(A)[I...] || throw(
        SpaceMismatch(
            "inserting a tensor of space $(space(v)) at $(I) into a SparseTensorArray with space $(eachspace(A))",
        ),
    )
    return nothing
end
