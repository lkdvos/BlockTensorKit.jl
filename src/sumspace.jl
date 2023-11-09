struct SumSpace{S<:ElementarySpace} <: ElementarySpace
    spaces::Vector{S}
end

SumSpace(spaces::S...) where {S<:ElementarySpace} = SumSpace(collect(spaces))
SumSpace{S}() where {S} = SumSpace(S[])

const ProductSumSpace{S,N} = ProductSpace{SumSpace{S},N} # for convenience
const TensorSumSpace{S} = TensorSpace{SumSpace{S}} # for convenience
const TensorMapSumSpace{S,N₁,N₂} = TensorMapSpace{SumSpace{S},N₁,N₂} # for convenience

TensorKit.InnerProductStyle(S::Type{<:SumSpace}) = InnerProductStyle(eltype(S))
TensorKit.sectortype(S::Type{<:SumSpace}) = sectortype(eltype(S))
TensorKit.field(::Type{SumSpace{S}}) where {S} = field(S)

Base.size(S::SumSpace) = size(S.spaces)

Base.getindex(S::SumSpace, i::Int) = S.spaces[i]
Base.getindex(S::SumSpace, i) = SumSpace(S.spaces[i])
Base.setindex!(S::SumSpace, args...) = setindex!(S.spaces, args...)

Base.lastindex(S::SumSpace) = lastindex(S.spaces)
Base.firstindex(S::SumSpace) = firstindex(S.spaces)

Base.eltype(V::SumSpace) = eltype(typeof(V))
Base.eltype(::Type{SumSpace{S}}) where {S} = S

Base.iterate(S::SumSpace, args...) = iterate(S.spaces, args...)

TensorKit.dims(S::SumSpace) = map(dim, S.spaces)
TensorKit.dim(S::SumSpace, n::Int) = dim(S.spaces[n])
TensorKit.dim(S::SumSpace) = sum(dims(S))

Base.axes(S::SumSpace) = Base.OneTo(dim(S))
Base.axes(S::SumSpace, n::Int) = axes(S.spaces, n)

Base.length(S::SumSpace) = length(S.spaces)

TensorKit.isdual(S::SumSpace) = isdual(first(S.spaces))
TensorKit.dual(S::SumSpace) = SumSpace(map(dual, S.spaces))
Base.conj(S::SumSpace) = dual(S)
TensorKit.flip(S::SumSpace) = SumSpace(map(flip, S.spaces))

function TensorKit.hassector(S::SumSpace, s::Sector)
    return mapreduce(v -> hassector(v, s), |, S.spaces; init=false)
end
function TensorKit.hassector(S::SumSpace, ::Trivial)
    return mapreduce(v -> hassector(v, Trivial()), |, S.spaces; init=false)
end

TensorKit.sectors(S::SumSpace) = TensorKit._sectors(S, sectortype(S))
function TensorKit._sectors(V::SumSpace, ::Type{Trivial})
    return TensorKit.TrivialOrEmptyIterator(dim(V) == 0)
end
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

Base.promote_rule(::Type{S}, ::Type{SumSpace{S}}) where {S} = SumSpace{S}
function Base.promote_rule(::Type{S1}, ::Type{<:ProductSpace{S2}}) where {S1<:ElementarySpace,S2<:ElementarySpace}
    return ProductSpace{promote_type(S1,S2)}
end
function Base.promote_rule(::Type{<:ProductSpace{S1}},
                           ::Type{<:ProductSpace{S2}}) where {S1<:ElementarySpace,
                                                              S2<:ElementarySpace}
    return ProductSpace{promote_type(S1,S2)}
end

Base.convert(::Type{I}, S::SumSpace{I}) where {I} = join(S)
Base.convert(::Type{SumSpace{S}}, V::S) where {S} = SumSpace(V)
function Base.convert(::Type{<:ProductSumSpace{S,N}}, V::ProductSpace{S,N}) where {S,N}
    return ProductSumSpace{S,N}(SumSpace.(V.spaces)...)
end
function Base.convert(::Type{<:ProductSumSpace{S}}, V::ProductSpace{S,N}) where {S,N}
    return ProductSumSpace{S,N}(SumSpace.(V.spaces)...)
end
function Base.convert(::Type{<:ProductSpace{S,N}}, V::ProductSumSpace{S,N}) where {S,N}
    return ProductSpace{S,N}(join.(V.spaces)...)
end

function Base.show(io::IO, V::SumSpace)
    if length(V) == 1
        print(io, "⨁(", V[1], ")")
    else
        print(io, "(")
        Base.join(io, V, " ⊕ ")
        print(io, ")")
    end
    return nothing
end

TensorKit.:⊕(S::ElementarySpace) = S
function TensorKit.:⊕(S1::SumSpace{I}, S2::SumSpace{I}) where {I}
    return SumSpace(vcat(S1.spaces, S2.spaces))
end

function TensorKit.fuse(V1::S, V2::S) where {S<:SumSpace}
    return SumSpace(vec([fuse(v1, v2) for (v1, v2) in Base.product(V1.spaces, V2.spaces)]))
end

Base.oneunit(S::Type{<:SumSpace}) = SumSpace(oneunit(eltype(S)))

sumspacetype(::Type{S}) where {S<:ElementarySpace} = SumSpace{S}

@inline Base.isassigned(S::SumSpace, i::Int) = isassigned(S.spaces, i)

# scalar indexing yields ProductSpace
function getsubspace(V::ProductSumSpace{S,N}, I::CartesianIndex{N}) where {S,N}
    return getsubspace(V, I.I...)
end
# ambiguity fix
function getsubspace(V::ProductSumSpace{S,1}, I::CartesianIndex{1}) where {S}
    return getsubspace(V, I.I...)
end
function getsubspace(V::ProductSumSpace{S,N}, I::Vararg{Int,N}) where {S,N}
    return ProductSpace{S,N}(map(getindex, V.spaces, I))
end

# non-scalar indexing yields ProductSumSpace
function getsubspace(V::ProductSumSpace{S,N}, I::Vararg{<:Any,N}) where {S,N}
    return ProductSumSpace{S,N}(map(getindex, V.spaces, I))
end

function getsubspace(V::TensorKit.HomSpace{<:SumSpace}, I::CartesianIndex{N}) where {N}
    N₁ = length(codomain(V))
    N₂ = length(domain(V))
    N₁ + N₂ == N || throw(ArgumentError("Invalid indexing"))
    return getsubspace(V.codomain, I.I[1:N₁]...) ← getsubspace(V.domain, I.I[(N₁ + 1):N]...)
end
function getsubspace(V::TensorKit.HomSpace{<:SumSpace}, I::Vararg{Int,N}) where {N}
    return getsubspace(V, CartesianIndex(I...))
end
function getsubspace(V::TensorKit.HomSpace{S,P₁,P₂}, I::Vararg{<:Any,N}) where {S,P₁,P₂,N}
    N₁ = length(codomain(V))
    N₂ = length(domain(V))
    N₁ + N₂ == N || throw(ArgumentError("Invalid indexing"))
    codomain′ = getsubspace(V.codomain, I[1:N₁]...)
    domain′ = getsubspace(V.domain, I[(N₁ + 1):N]...)
    return TensorKit.HomSpace{S,P₁,P₂}(codomain′, domain′)
end

function _getsubspace_scalar(V::ProductSumSpace{S,N}, I::Vararg{Int,N}) where {S,N}
    return ProductSpace{S,N}(map(getindex, V.spaces, I))
end
function _getsubspace_nonscalar(V::ProductSumSpace{S,N}, I::Vararg{<:Any,N}) where {S,N}
    return ProductSumSpace{S,N}(map(getindex, V.spaces, I))
end

function subblockdims(S::ProductSpace{<:SumSpace,N}, c::Sector) where {N}
    return N == 0 ? [1] :
           vec(map(I -> blockdim(getsubspace(S, I), c),
                   CartesianIndices(map(length, S.spaces))))
end
