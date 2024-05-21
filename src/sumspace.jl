"""
    struct SumSpace{S<:ElementarySpace} <: ElementarySpace

A (lazy) direct sum of elementary vector spaces of type `S`.
"""
struct SumSpace{S<:ElementarySpace} <: ElementarySpace
    spaces::Vector{S}
end

SumSpace(spaces::S...) where {S<:ElementarySpace} = SumSpace(collect(spaces))
SumSpace{S}() where {S} = SumSpace(S[])

# Convenience aliases
const ProductSumSpace{S,N} = ProductSpace{SumSpace{S},N}
const TensorSumSpace{S} = TensorSpace{SumSpace{S}}
const TensorMapSumSpace{S,N₁,N₂} = TensorMapSpace{SumSpace{S},N₁,N₂}

TensorKit.InnerProductStyle(S::Type{<:SumSpace}) = InnerProductStyle(eltype(S))
TensorKit.sectortype(S::Type{<:SumSpace}) = sectortype(eltype(S))
TensorKit.field(::Type{SumSpace{S}}) where {S} = field(S)

Base.size(S::SumSpace) = size(S.spaces)
Base.length(S::SumSpace) = length(S.spaces)

Base.getindex(S::SumSpace, i::Int) = S.spaces[i]
Base.getindex(S::SumSpace, i) = SumSpace(S.spaces[i])
Base.setindex!(S::SumSpace, args...) = setindex!(S.spaces, args...)

Base.iterate(S::SumSpace, args...) = iterate(S.spaces, args...)

Base.lastindex(S::SumSpace) = lastindex(S.spaces)
Base.firstindex(S::SumSpace) = firstindex(S.spaces)

Base.eltype(V::SumSpace) = eltype(typeof(V))
Base.eltype(::Type{SumSpace{S}}) where {S} = S

Base.axes(S::SumSpace) = Base.OneTo(dim(S))
Base.axes(S::SumSpace, n::Int) = axes(S.spaces, n)
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
@inline Base.isassigned(S::SumSpace, i::Int) = isassigned(S.spaces, i)

TensorKit.dims(S::SumSpace) = map(dim, S.spaces)
TensorKit.dim(S::SumSpace, n::Int) = dim(S.spaces[n])
TensorKit.dim(S::SumSpace) = sum(dims(S))

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
    return OneOrNoneIterator(dim(V) != 0, Trivial())
end
function TensorKit._sectors(S::SumSpace, ::Type{I}) where {I}
    s = Set{I}()
    for v in S.spaces
        s = s ∪ sectors(v)
    end
    return values(s)
end

TensorKit.dim(S::SumSpace, sector::Sector) = sum(v -> dim(v, sector), S.spaces; init=0)
# ambiguity fix:
TensorKit.dim(S::SumSpace, ::Trivial) = sum(v -> dim(v, Trivial()), S.spaces; init=0)

using TensorKit: ⊕

# TODO: find a better name for this function
function join(S::SumSpace)
    if length(S) == 1
        return only(S.spaces)
    else
        return ⊕(S.spaces...)
    end
end

# this conflicts with the definition in TensorKit, so users always need to specify
# ⊕(Vs::IndexSpace...) = SumSpace(Vs...)

Base.promote_rule(::Type{S}, ::Type{SumSpace{S}}) where {S} = SumSpace{S}
function Base.promote_rule(::Type{S1},
                           ::Type{<:ProductSpace{S2}}) where {S1<:ElementarySpace,
                                                              S2<:ElementarySpace}
    return ProductSpace{promote_type(S1, S2)}
end
function Base.promote_rule(::Type{<:ProductSpace{S1}},
                           ::Type{<:ProductSpace{S2}}) where {S1<:ElementarySpace,
                                                              S2<:ElementarySpace}
    return ProductSpace{promote_type(S1, S2)}
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

"""
    sumspacetype(::Union{S,Type{S}}) where {S<:ElementarySpace}

Return the type of a `SumSpace` with elements of type `S`.
"""
sumspacetype(::Type{S}) where {S<:ElementarySpace} = SumSpace{S}

@doc """
    getsubspace(V, I)

Return the subspace of `V` indexed by `I`.
""" getsubspace

# SumSpace indexing
# -----------------
# implementation very similar to CartesianIndices or LinearIndices

struct SumSpaceIndices{S,N₁,N₂,N} <:
       AbstractArray{HomSpace{S,ProductSpace{S,N₁},ProductSpace{S,N₂}},N}
    sumspaces::NTuple{N,SumSpace{S}}
    function SumSpaceIndices{S,N₁,N₂}(sumspaces::NTuple{N,SumSpace{S}}) where {S,N₁,N₂,N}
        @assert N == N₁ + N₂
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

Base.size(I::SumSpaceIndices) = map(length, I.sumspaces)

# simple scalar indexing
function Base.getindex(iter::SumSpaceIndices{S,N₁,N₂,N}, I::Vararg{Int,N}) where {S,N₁,N₂,N}
    codomain = ProductSpace{S,N₁}(map((inds, v) -> getindex(v.spaces, inds...), I[1:N₁],
                                      iter.sumspaces[1:N₁]))
    domain = ProductSpace{S,N₂}(map((inds, v) -> getindex(v.spaces, inds...),
                                    I[(N₁ + 1):N], iter.sumspaces[(N₁ + 1):N]))
    return HomSpace(codomain, domain)
end

# non-scalar indexing
@inline function Base._getindex(::IndexCartesian, iter::SumSpaceIndices{S,N₁,N₂,N},
                                I::Vararg{Union{Real,AbstractArray},N}) where {S,N₁,N₂,N}
    @boundscheck checkbounds(iter, I...)
    return SumSpaceIndices{S,N₁,N₂}(map(getindex, iter.sumspaces, I))
end

# disambiguation of base methods
function Base._getindex(::IndexCartesian, A::SumSpaceIndices{S,N₁,N₂,N},
                        I::Vararg{Int,M}) where {S,N₁,N₂,N,M}
    @inline
    @boundscheck checkbounds(A, I...) # generally _to_subscript_indices requires bounds checking
    @inbounds r = getindex(A, Base._to_subscript_indices(A, I...)...)
    return r
end
function Base._getindex(::IndexCartesian, A::SumSpaceIndices{S,N₁,N₂,N},
                        I::Vararg{Int,N}) where {S,N₁,N₂,N}
    Base.@_propagate_inbounds_meta
    return getindex(A, I...)
end

# # scalar indexing yields ProductSpace
# function getsubspace(V::ProductSumSpace{S,N}, I::CartesianIndex{N}) where {S,N}
#     return getsubspace(V, I.I...)
# end
# # ambiguity fix
# function getsubspace(V::ProductSumSpace{S,1}, I::CartesianIndex{1}) where {S}
#     return getsubspace(V, I.I...)
# end
# function getsubspace(V::ProductSumSpace{S,N}, I::Vararg{Int,N}) where {S,N}
#     return ProductSpace{S,N}(map(getindex, V.spaces, I))
# end

# # non-scalar indexing yields ProductSumSpace
# function getsubspace(V::ProductSumSpace{S,N}, I::Vararg{Any,N}) where {S,N}
#     return ProductSumSpace{S,N}(map(getindex, V.spaces, I))
# end

# function getsubspace(V::TensorKit.HomSpace{<:SumSpace}, I::CartesianIndex{N}) where {N}
#     N₁ = length(codomain(V))
#     N₂ = length(domain(V))
#     N₁ + N₂ == N || throw(ArgumentError("Invalid indexing"))
#     return getsubspace(V.codomain, I.I[1:N₁]...) ← getsubspace(V.domain, I.I[(N₁ + 1):N]...)
# end
# function getsubspace(V::TensorKit.HomSpace{<:SumSpace}, I::Vararg{Int,N}) where {N}
#     return getsubspace(V, CartesianIndex(I...))
# end
# function getsubspace(V::TensorKit.HomSpace{S,P₁,P₂}, I::Vararg{Any,N}) where {S,P₁,P₂,N}
#     N₁ = length(codomain(V))
#     N₂ = length(domain(V))
#     N₁ + N₂ == N || throw(ArgumentError("Invalid indexing"))
#     codomain′ = getsubspace(V.codomain, I[1:N₁]...)
#     domain′ = getsubspace(V.domain, I[(N₁ + 1):N]...)
#     return TensorKit.HomSpace{S,P₁,P₂}(codomain′, domain′)
# end

# function _getsubspace_scalar(V::ProductSumSpace{S,N}, I::Vararg{Int,N}) where {S,N}
#     return ProductSpace{S,N}(map(getindex, V.spaces, I))
# end
# function _getsubspace_nonscalar(V::ProductSumSpace{S,N}, I::Vararg{Any,N}) where {S,N}
#     return ProductSumSpace{S,N}(map(getindex, V.spaces, I))
# end

function subblockdims(V::ProductSumSpace{S,N}, c::Sector) where {S,N}
    return N == 0 ? [1] :
           vec(map(I -> blockdim(getsubspace(V, I), c),
                   CartesianIndices(map(length, V.spaces))))
end
