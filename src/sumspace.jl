struct SumSpace{𝕂,S<:ElementarySpace{𝕂}} <: ElementarySpace{𝕂}
    spaces::Vector{S}
end

SumSpace(spaces::S...) where {S <: ElementarySpace} = SumSpace([spaces...])
SumSpace{𝕂,S}() where {𝕂,S} = SumSpace(S())

TensorKit.InnerProductStyle(S::Type{<:SumSpace}) = InnerProductStyle(eltype(S))
TensorKit.sectortype(S::Type{<:SumSpace}) = sectortype(eltype(S))

Base.size(S::SumSpace) = size(S.spaces)

Base.getindex(S::SumSpace, i) = getindex(S.spaces, i...)
Base.setindex!(S::SumSpace, i) = setindex!(S.spaces, i...)

Base.eltype(::SumSpace{<:Any,S}) where {S} = S
Base.eltype(::Type{<:SumSpace{<:Any,S}}) where {S} = S

Base.iterate(S::SumSpace, args...) = iterate(S.spaces, args...)


TensorKit.dims(S::SumSpace) = map(dim, S.spaces)
TensorKit.dim(S::SumSpace, n::Int) = dim(S.spaces[n])
TensorKit.dim(S::SumSpace) = sum(dims(S))

Base.axes(S::SumSpace) = Base.OneTo(dim(S))
Base.axes(S::SumSpace, n::Int) = axes(S.spaces, n)

Base.length(S::SumSpace) = length(S.spaces)

TensorKit.isdual(S::SumSpace) = isdual(first(S.spaces))
TensorKit.dual(S::SumSpace) = SumSpace(map(dual, S.spaces))
TensorKit.flip(S::SumSpace) = SumSpace(map(flip, S.spaces))

function TensorKit.hassector(S::SumSpace, s::Sector)
    return mapreduce(v -> hassector(v, s), |, S.spaces; init=false)
end
TensorKit.hassector(S::SumSpace, ::Trivial) = mapreduce(v -> hassector(v, Trivial()), |, S.spaces; init=false)

TensorKit.sectors(S::SumSpace) = TensorKit._sectors(S, sectortype(S))
TensorKit._sectors(V::SumSpace, ::Type{Trivial}) = TensorKit.TrivialOrEmptyIterator(dim(V) == 0)
TensorKit._sectors(S::SumSpace, ::Type{<:Sector}) = union(map(sectors, S.spaces))

TensorKit.dim(S::SumSpace, sector::Sector) = mapreduce(v -> dim(v, sector), +, S.spaces)
TensorKit.dim(S::SumSpace, ::Trivial) = mapreduce(v -> dim(v, Trivial()), +, S.spaces)

function Base.axes(S::SumSpace, c::Sector)
    offset = 0
    a = []
    for s in S.spaces
        a = push!(a, axes(s, c) .+ offset)
        offset += dim(s)
    end
    return collect(flatten(a))
end

Base.hash(S::SumSpace, h::UInt) = hash(S.spaces, h)
Base.:(==)(S1::SumSpace, S2::SumSpace) = S1.spaces == S2.spaces

using TensorKit: ⊕

function join(S::SumSpace)
    if length(S) == 1
        return only(S.spaces)
    else
        return ⊕(S.spaces...)
    end
end

Base.convert(::Type{I}, S::SumSpace{<:Any,I}) where {I} = join(S)

TensorKit.:⊕(S::ElementarySpace) = S
function TensorKit.:⊕(S1::SumSpace{I}, S2::SumSpace{I}) where {I}
    return SumSpace(vcat(S1.spaces, S2.spaces))
end

TensorKit.fuse(V1::S, V2::S) where {S<:SumSpace} = SumSpace(fuse(⊕(V1.spaces...), ⊕(V2.spaces...)))



Base.oneunit(S::Type{<:SumSpace}) = SumSpace(oneunit(eltype(S)))

sumspacetype(::Type{S}) where {S<:ElementarySpace} = SumSpace{field(S),S}

@inline Base.isassigned(S::SumSpace, i::Int) = isassigned(S.spaces, i)

function getsubspace(V::ProductSpace{S,N}, I::CartesianIndex{N}) where {S<:SumSpace,N}
    return ProductSpace{eltype(S),N}(map(getindex, V.spaces, I.I))
end

getsubspace(S::ProductSpace{<:SumSpace,N}, I::Vararg{Int,N}) where {N} = getsubspace(S, CartesianIndex(I...))

function getsubspace(V::TensorKit.HomSpace{S,P1,P2}, I::CartesianIndex{N}) where {S<:SumSpace,N₁,N₂,P1<:ProductSpace{S,N₁},P2<:ProductSpace{S,N₂},N}
    N₁ + N₂ == N || throw(ArgumentError())
    return getsubspace(V.codomain, I.I[1:N₁]...) ← getsubspace(V.domain, I.I[N₁+1:N]...)
end
getsubspace(V::TensorKit.HomSpace{S,P1,P2}, I::Vararg{Int,N}) where {S<:SumSpace,N₁,N₂,P1<:ProductSpace{S,N₁},P2<:ProductSpace{S,N₂},N} = getsubspace(V, CartesianIndex(I...))




function subblockdims(S::ProductSpace{<:SumSpace,N}, c::Sector) where {N}
    return N == 0 ? [1] : vec(map(I -> blockdim(getsubspace(S, I), c), CartesianIndices(map(length, S.spaces))))
end