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

# AbstractArray behavior
# ----------------------
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

# VectorSpace behavior
# --------------------
TensorKit.InnerProductStyle(S::Type{<:SumSpace}) = InnerProductStyle(eltype(S))
TensorKit.sectortype(S::Type{<:SumSpace}) = sectortype(eltype(S))
TensorKit.field(::Type{SumSpace{S}}) where {S} = field(S)

"""
    sumspacetype(::Union{S,Type{S}}) where {S<:ElementarySpace}

Return the type of a `SumSpace` with elements of type `S`.
"""
sumspacetype(::Type{S}) where {S<:ElementarySpace} = SumSpace{S}

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

TensorKit.compose(V, W) = TensorKit.compose(promote(V, W)...)

# this conflicts with the definition in TensorKit, so users always need to specify
# ⊕(Vs::IndexSpace...) = SumSpace(Vs...)

TensorKit.:⊕(S::ElementarySpace) = S
function TensorKit.:⊕(S1::SumSpace{I}, S2::SumSpace{I}) where {I}
    return SumSpace(vcat(S1.spaces, S2.spaces))
end

function TensorKit.fuse(V1::S, V2::S) where {S<:SumSpace}
    return SumSpace(vec([fuse(v1, v2) for (v1, v2) in Base.product(V1.spaces, V2.spaces)]))
end

Base.oneunit(S::Type{<:SumSpace}) = SumSpace(oneunit(eltype(S)))

# Promotion and conversion
# ------------------------
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
function Base.promote_rule(::Type{<:TensorMapSumSpace{S}},
                           ::Type{<:TensorMapSpace{S}}) where {S}
    return TensorMapSumSpace{S}
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
function Base.convert(::Type{<:TensorMapSumSpace{S}},
                      V::TensorMapSpace{S,N₁,N₂}) where {S,N₁,N₂}
    return convert(ProductSumSpace{S,N₁}, codomain(V)) ←
           convert(ProductSumSpace{S,N₂}, domain(V))
end

# Show
# ----
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
