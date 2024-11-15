"""
    struct SparseBlockTensorMap{TT<:AbstractTensorMap{E,S,N₁,N₂}} <: AbstractBlockTensorMap{E,S,N₁,N₂}

Sparse `SparseBlockTensorMap` type that stores tensors of type `TT` in a sparse dictionary.
"""
struct SparseBlockTensorMap{TT<:AbstractTensorMap,E,S,N₁,N₂,N} <:
       AbstractBlockTensorMap{E,S,N₁,N₂}
    data::Dict{CartesianIndex{N},TT}
    space::TensorMapSumSpace{S,N₁,N₂}

    # uninitialized constructor
    function SparseBlockTensorMap{TT,E,S,N₁,N₂,N}(
        ::UndefBlocksInitializer, space::TensorMapSumSpace{S,N₁,N₂}
    ) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
        @assert N₁ + N₂ == N "SparseBlockTensorMap: data has wrong number of dimensions"
        data = Dict{CartesianIndex{N},TT}()
        return new{TT,E,S,N₁,N₂,N}(data, space)
    end

    # constructor from data
    function SparseBlockTensorMap{TT,E,S,N₁,N₂,N}(
        data::Dict{CartesianIndex{N},TT}, space::TensorMapSumSpace{S,N₁,N₂}
    ) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
        @assert N₁ + N₂ == N "SparseBlockTensorMap: data has wrong number of dimensions"
        return new{TT,E,S,N₁,N₂,N}(data, space)
    end
end

function SparseBlockTensorMap{TT,E,S,N₁,N₂,N}(
    ::UndefInitializer, space::TensorMapSumSpace{S,N₁,N₂}
) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    return SparseBlockTensorMap{TT,E,S,N₁,N₂,N}(undef_blocks, space)
end

# uninitialized constructor
function SparseBlockTensorMap{TT}(
    ::Union{UndefBlocksInitializer,UndefInitializer}, space::TensorMapSumSpace{S,N₁,N₂}
) where {E,S,N₁,N₂,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    N = N₁ + N₂
    return SparseBlockTensorMap{TT,E,S,N₁,N₂,N}(undef_blocks, space)
end

# constructor from data
function SparseBlockTensorMap{TT}(
    data::Dict{CartesianIndex{N},TT}, space::TensorMapSumSpace{S,N₁,N₂}
) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    return SparseBlockTensorMap{TT,E,S,N₁,N₂,N}(data, space)
end

function sparseblocktensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
    TT = tensormaptype(S, N₁, N₂, T)
    return SparseBlockTensorMap{TT}
end
function sparseblocktensormaptype(
    ::Type{SumSpace{S}}, N₁::Int, N₂::Int, ::Type{T}
) where {S,T}
    TT = tensormaptype(S, N₁, N₂, T)
    return SparseBlockTensorMap{TT}
end

# Constructors
# ------------
function SparseBlockTensorMap{TT}(
    data::Union{Array{TT},UndefInitializer,UndefBlocksInitializer},
    codom::ProductSumSpace,
    dom::ProductSumSpace,
) where {TT}
    return SparseBlockTensorMap{TT}(data, codom ← dom)
end

# AbstractBlockTensorMap -> SparseBlockTensorMap
function SparseBlockTensorMap(t::AbstractBlockTensorMap)
    t isa SparseBlockTensorMap && return t # TODO: should this copy?
    tdst = SparseBlockTensorMap{eltype(t)}(undef_blocks, space(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = v
    end
    return tdst
end

# AbstractTensorMap -> SparseBlockTensorMap
function SparseBlockTensorMap(t::AbstractTensorMap, space::TensorMapSumSpace)
    TT = tensormaptype(spacetype(t), numout(t), numin(t), storagetype(t))
    tdst = SparseBlockTensorMap{TT}(undef, space)
    for (f₁, f₂) in fusiontrees(tdst)
        tdst[f₁, f₂] = t[f₁, f₂]
    end
    return tdst
end

# Utility constructors
# --------------------

sprand(V::VectorSpace, p::Real) = sprand(Float64, V, p)
function sprand(::Type{T}, V::TensorMapSumSpace, p::Real) where {T<:Number}
    TT = sparseblocktensormaptype(spacetype(V), numout(V), numin(V), T)
    t = TT(undef, V)
    for (I, v) in enumerate(eachspace(t))
        if rand() < p
            t[I] = rand(T, v)
        end
    end
    return t
end
function sprand(::Type{T}, V::VectorSpace, p::Real) where {T<:Number}
    return sprand(T, V ← one(V), p)
end

# Properties
# ----------
TK.space(t::SparseBlockTensorMap) = t.space
VI.scalartype(::Type{<:SparseBlockTensorMap{TT}}) where {TT} = scalartype(TT)

Base.parent(t::SparseBlockTensorMap) = SparseTensorArray(t.data, space(t))
Base.eltype(::Type{<:SparseBlockTensorMap{TT}}) where {TT} = TT

issparse(::SparseBlockTensorMap) = true
nonzero_keys(t::SparseBlockTensorMap) = keys(t.data)
nonzero_values(t::SparseBlockTensorMap) = values(t.data)
nonzero_pairs(t::SparseBlockTensorMap) = pairs(t.data)
nonzero_length(t::SparseBlockTensorMap) = length(t.data)

# Utility
# -------
function Base.delete!(t::SparseBlockTensorMap, I::CartesianIndex)
    delete!(t.data, I)
    return t
end
function Base.delete!(t::SparseBlockTensorMap{TT}, I::Vararg{Int,N}) where {TT,N}
    return delete!(t, CartesianIndex(I...))
end

# Show
# ----
function Base.summary(io::IO, t::SparseBlockTensorMap)
    szstring = Base.dims2string(size(t))
    TT = eltype(t)
    typeinfo = get(io, :typeinfo, Any)
    if typeinfo <: typeof(t) || typeinfo <: TT
        typestring = ""
    else
        typestring = "{$TT}"
    end
    V = space(t)
    return print(io, "$szstring SparseBlockTensorMap$typestring($V)")
end
