# SumSpace indexing
# -----------------
# implementation very similar to CartesianIndices or LinearIndices

"""
    eachspace(V::TensorMapSumSpace) -> SumSpaceIndices

Return an object that can be used to obtain the subspaces of a `BlockTensorMap`.
"""
eachspace(V::TensorMapSumSpace) = SumSpaceIndices(V)

"""
    struct SumSpaceIndices{S,N₁,N₂} <: AbstractArray{TensorMapSpace{S,N₁,N₂},N₁ + N₂}
"""
struct SumSpaceIndices{S,N₁,N₂,N} <: AbstractArray{TensorMapSpace{S,N₁,N₂},N}
    sumspaces::NTuple{N,SumSpace{S}}
    function SumSpaceIndices{S,N₁,N₂}(sumspaces::NTuple{N,SumSpace{S}}) where {S,N₁,N₂,N}
        @assert N == N₁ + N₂ "Invalid number of spaces"
        return new{S,N₁,N₂,N}(sumspaces)
    end
end
function SumSpaceIndices(V::HomSpace{SumSpace{S}}) where {S}
    N₁ = length(codomain(V))
    N₂ = length(domain(V))
    return SumSpaceIndices{S,N₁,N₂}((V.codomain..., V.domain...))
end
function SumSpaceIndices{S,N₁,N₂}(spaces::Tuple) where {S,N₁,N₂}
    return SumSpaceIndices{S,N₁,N₂}(map(x -> convert(SumSpace{S}, x), spaces))
end

# Overload show of type to hide the inferred last type parameter
function Base.show(io::IO, ::Type{<:SumSpaceIndices{S,N₁,N₂}}) where {S,N₁,N₂}
    return print(io, "SumSpaceIndices{", S, ",", N₁, ",", N₂, "}")
end

Base.size(I::SumSpaceIndices) = map(length, I.sumspaces)
Base.IndexStyle(::Type{<:SumSpaceIndices}) = IndexCartesian()

# simple scalar indexing
function Base.getindex(iter::SumSpaceIndices{S,N₁,N₂,N}, I::Vararg{Int,N}) where {S,N₁,N₂,N}
    codomain = ProductSpace{S,N₁}(
        map((inds, v) -> getindex(v.spaces, inds...), I[1:N₁], iter.sumspaces[1:N₁])
    )
    domain = ProductSpace{S,N₂}(
        map(
            (inds, v) -> getindex(v.spaces, inds...),
            I[(N₁ + 1):N],
            iter.sumspaces[(N₁ + 1):N],
        ),
    )
    return HomSpace(codomain, domain)
end

# non-scalar indexing
@inline function Base._getindex(
    ::IndexCartesian,
    iter::SumSpaceIndices{S,N₁,N₂,N},
    I::Vararg{Union{Real,AbstractArray},N},
) where {S,N₁,N₂,N}
    @boundscheck checkbounds(iter, I...)
    return SumSpaceIndices{S,N₁,N₂}(map(getindex, iter.sumspaces, I))
end
@inline function Base._getindex(
    ::IndexCartesian, iter::SumSpaceIndices{S,N₁,N₂}, I::Union{Real,AbstractVector}
) where {S,N₁,N₂}
    @boundscheck checkbounds(iter, I)
    nontrivial_sizes = findall(>(1), size(iter))
    if isempty(nontrivial_sizes)
        I′ = ntuple(i -> i == 1 ? I : 1, ndims(iter))
    elseif length(nontrivial_sizes) == 1
        I′ = ntuple(i -> i in nontrivial_sizes ? I : 1, ndims(iter))
    else
        throw(ArgumentError("Cannot index $iter with $I"))
    end
    return Base._getindex(IndexCartesian(), iter, I′...)
end
# disambiguate:
@inline function Base._getindex(
    ::IndexCartesian, iter::SumSpaceIndices{S,N₁,N₂,1}, I::Union{Real,AbstractVector}
) where {S,N₁,N₂}
    @boundscheck checkbounds(iter, I)
    return SumSpaceIndices{S,N₁,N₂}(map(getindex, iter.sumspaces, (I,)))
end

@inline Base._getindex(::IndexCartesian, iter::SumSpaceIndices, ::Colon) = iter

# disambiguation of base methods
function Base._getindex(
    ::IndexCartesian, A::SumSpaceIndices{S,N₁,N₂,N}, I::Vararg{Int,M}
) where {S,N₁,N₂,N,M}
    @inline
    @boundscheck checkbounds(A, I...) # generally _to_subscript_indices requires bounds checking
    @inbounds r = getindex(A, Base._to_subscript_indices(A, I...)...)
    return r
end
function Base._getindex(
    ::IndexCartesian, A::SumSpaceIndices{S,N₁,N₂,N}, I::Vararg{Int,N}
) where {S,N₁,N₂,N}
    Base.@_propagate_inbounds_meta
    return getindex(A, I...)
end

function TensorKit.space(I::SumSpaceIndices{S,N₁,N₂,N}) where {S,N₁,N₂,N}
    cod = prod(I.sumspaces[1:N₁]; init=one(sumspacetype(S)))
    dom = prod(I.sumspaces[(N₁ + 1):end]; init=one(sumspacetype(S)))
    return cod ← dom
end

function subblockdims(V::ProductSumSpace{S,N}, c::Sector) where {S,N}
    return if N == 0
        [1]
    else
        vec(
            map(
                I -> blockdim(getsubspace(V, I), c), CartesianIndices(map(length, V.spaces))
            ),
        )
    end
end

function Base._cat(dims, A::SumSpaceIndices{S,N₁,N₂}...) where {S,N₁,N₂}
    @assert maximum(dims) <= N₁ + N₂ "Invalid number of spaces"
    catdims = Base.dims2cat(dims)
    Vs = ntuple(N₁ + N₂) do i
        return if i <= length(catdims) && catdims[i]
            ⊕((A[j].sumspaces[i] for j in 1:length(A))...)
        else
            A[1].sumspaces[i]
        end
    end
    return SumSpaceIndices{S,N₁,N₂}(Vs)
end
