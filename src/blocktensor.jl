"""
    struct BlockTensorMap{TT<:AbstractTensorMap{E,S,N₁,N₂}} <: AbstractTensorMap{E,S,N₁,N₂}

Dense `BlockTensorMap` type that stores tensors of type `TT` in a dense array.
"""
struct BlockTensorMap{TT<:AbstractTensorMap,E,S,N₁,N₂,N} <:
       AbstractBlockTensorMap{E,S,N₁,N₂}
    data::Array{TT,N}
    codom::ProductSumSpace{S,N₁}
    dom::ProductSumSpace{S,N₂}

    # constructor from data
    function BlockTensorMap{TT}(
        data::Array{TT,N}, codom::ProductSumSpace{S,N₁}, dom::ProductSumSpace{S,N₂}
    ) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
        @assert N₁ + N₂ == N "BlockTensorMap: data has wrong number of dimensions"
        return new{TT,E,S,N₁,N₂,N}(data, codom, dom)
    end
end

# hack to avoid too many type parameters, which are enforced by inner constructor
function Base.show(io::IO, ::Type{TT}) where {TT<:BlockTensorMap}
    return print(io, "BlockTensorMap{", eltype(TT), "}")
end
function Base.show(io::IO, ::Type{BlockTensorMap})
    return print(io, "BlockTensorMap")
end

function blocktensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
    TT = tensormaptype(S, N₁, N₂, T)
    return BlockTensorMap{TT}
end
function blocktensormaptype(::Type{SumSpace{S}}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
    TT = tensormaptype(S, N₁, N₂, T)
    return BlockTensorMap{TT}
end

# Undef constructors
# ------------------
function BlockTensorMap{TT}(
    ::UndefBlocksInitializer, codom::ProductSumSpace{S,N₁}, dom::ProductSumSpace{S,N₂}
) where {E,S,N₁,N₂,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    N = N₁ + N₂
    data = Array{TT,N}(undef, size(SumSpaceIndices(codom ← dom)))
    return BlockTensorMap{TT}(data, codom, dom)
end

function BlockTensorMap{TT}(
    ::UndefInitializer, codom::ProductSumSpace{S,N₁}, dom::ProductSumSpace{S,N₂}
) where {E,S,N₁,N₂,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    # preallocate data to ensure correct eltype
    data = Array{TT,N₁ + N₂}(undef, size(SumSpaceIndices(codom ← dom)))
    map!(Base.Fix1(similar, TT), data, SumSpaceIndices(codom ← dom))
    return BlockTensorMap{TT}(data, codom, dom)
end

function BlockTensorMap{TT}(
    ::Union{UndefInitializer,UndefBlocksInitializer}, V::TensorMapSumSpace{S,N₁,N₂}
) where {E,S,N₁,N₂,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    return BlockTensorMap{TT}(undef, codomain(V), domain(V))
end

# Convenience constructors
# ------------------------
for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function Base.$fname(::Type{T}, V::TensorMapSumSpace) where {T}
            TT = blocktensormaptype(spacetype(V), numout(V), numin(V), T)
            t = TT(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randfun in (:rand, :randn, :randexp)
    randfun! = Symbol(randfun, :!)
    @eval begin
        function Random.$randfun(
            rng::Random.AbstractRNG, ::Type{T}, V::TensorMapSumSpace
        ) where {T}
            TT = blocktensormaptype(spacetype(V), numout(V), numin(V), T)
            t = TT(undef, V)
            Random.$randfun!(rng, t)
            return t
        end

        function Random.$randfun!(rng::Random.AbstractRNG, t::BlockTensorMap)
            foreach(b -> Random.$randfun!(rng, b), parent(t))
            return t
        end
    end
end

function BlockTensorMap(t::AbstractBlockTensorMap)
    t isa BlockTensorMap && return t # TODO: should this copy?
    tdst = BlockTensorMap{eltype(t)}(undef_blocks, codomain(t), domain(t))
    for I in eachindex(t)
        tdst[I] = t[I]
    end
    return tdst
end

# Properties
# ----------
Base.eltype(::Type{<:BlockTensorMap{TT}}) where {TT} = TT
Base.parent(t::BlockTensorMap) = t.data

function Base.copyto!(
    dest::BlockTensorMap,
    Rdest::CartesianIndices,
    src::BlockTensorMap,
    Rsrc::CartesianIndices,
)
    copyto!(parent(dest), Rdest, parent(src), Rsrc)
    return dest
end

TK.codomain(t::BlockTensorMap) = t.codom
TK.domain(t::BlockTensorMap) = t.dom

issparse(::BlockTensorMap) = false

# Utility
# -------

Base.delete!(t::BlockTensorMap, I...) = (zerovector!(getindex(t, I...)); t)

# Show
# ----
function Base.summary(io::IO, t::BlockTensorMap)
    szstring = Base.dims2string(size(t))
    TT = eltype(t)
    V = space(t)
    return print(io, "$szstring BlockTensorMap{$TT}($V)")
end

# Converters
# ----------

function Base.promote_rule(
    ::Type{<:BlockTensorMap{TT₁}}, ::Type{<:BlockTensorMap{TT₂}}
) where {TT₁,TT₂}
    TT = promote_type(TT₁, TT₂)
    return BlockTensorMap{TT}
end

function Base.convert(::Type{<:BlockTensorMap{TT₁}}, t::BlockTensorMap{TT₂}) where {TT₁,TT₂}
    TT₁ === TT₂ && return t
    tdst = BlockTensorMap{TT₁}(undef, codomain(t), domain(t))
    for I in eachindex(t)
        tdst[I] = t[I]
    end
    return tdst
end

function Base.convert(::Type{BlockTensorMap}, t::AbstractTensorMap)
    t isa BlockTensorMap && return t
    S = spacetype(t)
    N₁ = numout(t)
    N₂ = numin(t)
    TT = blocktensormaptype(S, N₁, N₂, storagetype(t))
    tdst = TT(
        undef,
        convert(ProductSumSpace{S,N₁}, codomain(t)),
        convert(ProductSumSpace{S,N₂}, domain(t)),
    )
    tdst[1] = t
    return tdst
end

function Base.convert(::Type{TT}, t::BlockTensorMap) where {TT<:BlockTensorMap}
    t isa TT && return t

    tdst = TT(undef, space(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = v
    end
    return tdst
end

# Utility
# -------
function Base.copy(tsrc::BlockTensorMap{E,S,N1,N2,N}) where {E,S,N1,N2,N}
    tdst = similar(tsrc)
    for (key, value) in nonzero_pairs(tsrc)
        tdst[key] = copy(value)
    end
    return tdst
end

Base.haskey(t::BlockTensorMap, I::CartesianIndex) = haskey(t.data, I)
function Base.haskey(t::BlockTensorMap, i::Int)
    return haskey(t.data, CartesianIndices(t)[i])
end
