"""
    struct BlockTensorMap{TT<:AbstractTensorMap{E,S,N₁,N₂}} <: AbstractTensorMap{E,S,N₁,N₂}

Dense `BlockTensorMap` type that stores tensors of type `TT` in a dense array.
"""
struct BlockTensorMap{TT<:AbstractTensorMap,E,S,N₁,N₂,N} <:
       AbstractBlockTensorMap{E,S,N₁,N₂}
    data::Array{TT,N}
    space::TensorMapSumSpace{S,N₁,N₂}

    # uninitialized constructor
    function BlockTensorMap{TT,E,S,N₁,N₂,N}(
        ::UndefBlocksInitializer, space::TensorMapSumSpace{S,N₁,N₂}
    ) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
        @assert N₁ + N₂ == N "BlockTensorMap: data has wrong number of dimensions"
        data = Array{TT,N}(undef, size(SumSpaceIndices(space)))
        return new{TT,E,S,N₁,N₂,N}(data, space)
    end

    # constructor from data
    function BlockTensorMap{TT,E,S,N₁,N₂,N}(
        data::Array{TT,N}, space::TensorMapSumSpace{S,N₁,N₂}
    ) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
        @assert N₁ + N₂ == N "BlockTensorMap: data has wrong number of dimensions"
        return new{TT,E,S,N₁,N₂,N}(data, space)
    end
end

function BlockTensorMap{TT,E,S,N₁,N₂,N}(
    ::UndefInitializer, space::TensorMapSumSpace{S,N₁,N₂}
) where {TT,E,S,N₁,N₂,N}
    tdst = BlockTensorMap{TT,E,S,N₁,N₂,N}(undef_blocks, space)
    tdst.data .= similar.(TT, SumSpaceIndices(space))
    return tdst
end

# uninitialized constructor
function BlockTensorMap{TT}(
    u::Union{UndefBlocksInitializer,UndefInitializer}, space::TensorMapSumSpace{S,N₁,N₂}
) where {E,S,N₁,N₂,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    N = N₁ + N₂
    return BlockTensorMap{TT,E,S,N₁,N₂,N}(u, space)
end

# constructor from data
function BlockTensorMap{TT}(
    data::Array{TT,N}, space::TensorMapSumSpace{S,N₁,N₂}
) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    @assert N₁ + N₂ == N "BlockTensorMap: data has wrong number of dimensions"
    return BlockTensorMap{TT,E,S,N₁,N₂,N}(data, space)
end

function blocktensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
    TT = tensormaptype(S, N₁, N₂, T)
    return BlockTensorMap{TT}
end
function blocktensormaptype(::Type{SumSpace{S}}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
    TT = tensormaptype(S, N₁, N₂, T)
    return BlockTensorMap{TT}
end

# Constructors
# ------------
function BlockTensorMap{TT}(
    data::Union{Array{TT},UndefInitializer,UndefBlocksInitializer},
    codom::ProductSumSpace{S,N₁},
    dom::ProductSumSpace{S,N₂},
) where {TT,S,N₁,N₂}
    return BlockTensorMap{TT}(data, codom ← dom)
end

function BlockTensorMap(
    f::Union{UndefInitializer,UndefBlocksInitializer}, space::TensorMapSumSpace{S,N₁,N₂}
) where {S,N₁,N₂}
    TT = tensormaptype(S, N₁, N₂, Float64)
    return BlockTensorMap{TT}(f, space)
end
function BlockTensorMap(
    f::Union{UndefInitializer,UndefBlocksInitializer},
    codom::ProductSumSpace,
    dom::ProductSumSpace,
)
    return BlockTensorMap(f, codom ← dom)
end

function BlockTensorMap(
    data::Array{TT}, space::TensorMapSumSpace{S,N₁,N₂}
) where {S,N₁,N₂,TT<:AbstractTensorMap{<:Any,S,N₁,N₂}}
    return BlockTensorMap{TT}(data, space)
end
function BlockTensorMap(
    data::Array{<:AbstractTensorMap}, codom::ProductSumSpace, dom::ProductSumSpace
)
    return BlockTensorMap(data, codom ← dom)
end

# AbstractBlockTensorMap -> BlockTensorMap
function BlockTensorMap(t::AbstractBlockTensorMap)
    t isa BlockTensorMap && return t # TODO: should this copy?
    tdst = BlockTensorMap{eltype(t)}(undef_blocks, space(t))
    for I in eachindex(t)
        tdst[I] = t[I]
    end
    return tdst
end

# AbstractTensorMap -> BlockTensorMap
function BlockTensorMap(t::AbstractTensorMap, space::TensorMapSumSpace)
    TT = tensormaptype(spacetype(t), numout(t), numin(t), storagetype(t))
    tdst = BlockTensorMap{TT}(undef, space)
    for (f₁, f₂) in fusiontrees(tdst)
        tdst[f₁, f₂] .= t[f₁, f₂]
    end
    return tdst
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

TK.space(t::BlockTensorMap) = t.space
VI.scalartype(::Type{<:BlockTensorMap{TT}}) where {TT} = scalartype(TT)

issparse(::BlockTensorMap) = false

# Utility
# -------
Base.delete!(t::BlockTensorMap, I...) = (zerovector!(getindex(t, I...)); t)

# Show
# ----
function Base.summary(io::IO, t::BlockTensorMap)
    szstring = Base.dims2string(size(t))
    TT = eltype(t)
    typeinfo = get(io, :typeinfo, Any)
    if typeinfo <: typeof(t) || typeinfo <: TT
        typestring = ""
    else
        typestring = "{$TT}"
    end
    V = space(t)
    return print(io, "$szstring BlockTensorMap$typestring($V)")
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
    tdst = BlockTensorMap{TT₁}(undef, space(t))
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

    tdst = TT(undef_blocks, space(t))
    for I in CartesianIndices(t)
        tdst[I] = t[I]
    end
    return tdst
end

# Utility
# -------
Base.haskey(t::BlockTensorMap, I::CartesianIndex) = haskey(t.data, I)
function Base.haskey(t::BlockTensorMap, i::Int)
    return haskey(t.data, CartesianIndices(t)[i])
end
