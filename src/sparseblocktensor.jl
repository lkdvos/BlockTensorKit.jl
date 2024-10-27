"""
    struct SparseBlockTensorMap{TT<:AbstractTensorMap{E,S,N₁,N₂}} <: AbstractBlockTensorMap{E,S,N₁,N₂}

Sparse `SparseBlockTensorMap` type that stores tensors of type `TT` in a sparse dictionary.
"""
struct SparseBlockTensorMap{TT<:AbstractTensorMap,E,S,N₁,N₂,N} <:
       AbstractBlockTensorMap{E,S,N₁,N₂}
    data::Dict{CartesianIndex{N},TT}
    codom::ProductSumSpace{S,N₁}
    dom::ProductSumSpace{S,N₂}

    function SparseBlockTensorMap{TT}(
        data::Dict{CartesianIndex{N},TT},
        codom::ProductSumSpace{S,N₁},
        dom::ProductSumSpace{S,N₂},
    ) where {E,S,N₁,N₂,N,TT<:AbstractTensorMap{E,S,N₁,N₂}}
        @assert N₁ + N₂ == N "SparseBlockTensorMap: data has wrong number of dimensions"
        return new{TT,E,S,N₁,N₂,N}(data, codom, dom)
    end
end

# hack to avoid too many type parameters, enforced by inner constructor
# function Base.show(io::IO, ::Type{TT}) where {TT<:SparseBlockTensorMap}
#     return print(io, "SparseBlockTensorMap{", eltype(TT), "}")
# end
# function Base.show(io::IO, ::Type{SparseBlockTensorMap})
#     return print(io, "SparseBlockTensorMap")
# end

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
function sparseblocktensormaptype(
    TT::Type{<:AbstractTensorMap{E,S}}, N₁::Int, N₂::Int, ::Type{T}
) where {E,S,T}
    TT′ = isabstracttype(TT) ? AbstractTensorMap{E,S,N₁,N₂} : tensormaptype(S, N₁, N₂, T)
    return SparseBlockTensorMap{TT′}
end
# Undef constructors
# ------------------
# no difference between UndefInitializer and UndefBlocksInitializer
function SparseBlockTensorMap{TT}(
    ::Union{UndefBlocksInitializer,UndefInitializer},
    codom::ProductSumSpace{S,N₁},
    dom::ProductSumSpace{S,N₂},
) where {E,S,N₁,N₂,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    N = N₁ + N₂
    data = Dict{CartesianIndex{N},TT}()
    return SparseBlockTensorMap{TT}(data, codom, dom)
end

function SparseBlockTensorMap{TT}(
    ::Union{UndefInitializer,UndefBlocksInitializer}, V::TensorMapSumSpace{S,N₁,N₂}
) where {E,S,N₁,N₂,TT<:AbstractTensorMap{E,S,N₁,N₂}}
    return SparseBlockTensorMap{TT}(undef, codomain(V), domain(V))
end

# Utility constructors
# --------------------
function SparseBlockTensorMap(t::AbstractBlockTensorMap)
    t isa SparseBlockTensorMap && return t # TODO: should this copy?
    tdst = SparseBlockTensorMap{eltype(t)}(undef_blocks, codomain(t), domain(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = v
    end
    return tdst
end

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

# specific implementation for SparseBlockTensorMap with Sumspace -> returns `SparseBlockTensorMap`
function Base.similar(
    ::SparseBlockTensorMap{TT}, TorA::Type, P::TensorMapSumSpace{S}
) where {TT,S}
    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    TT′ = sparseblocktensormaptype(TT, N₁, N₂, TorA)
    return TT′(undef, codomain(P), domain(P))
end
function Base.similar(::Type{<:SparseBlockTensorMap{TT}}, P::TensorMapSumSpace) where {TT}
    return SparseBlockTensorMap{TT}(undef, codomain(P), domain(P))
end
# Properties
# ----------
TensorKit.domain(t::SparseBlockTensorMap) = t.dom
TensorKit.codomain(t::SparseBlockTensorMap) = t.codom

Base.parent(t::SparseBlockTensorMap) = SparseTensorArray(t.data, space(t))
Base.eltype(::Type{<:SparseBlockTensorMap{TT}}) where {TT} = TT

issparse(::SparseBlockTensorMap) = true
nonzero_keys(t::SparseBlockTensorMap) = keys(t.data)
nonzero_values(t::SparseBlockTensorMap) = values(t.data)
nonzero_pairs(t::SparseBlockTensorMap) = pairs(t.data)
nonzero_length(t::SparseBlockTensorMap) = length(t.data)

function dropzeros!(t::SparseBlockTensorMap)
    for (k, v) in nonzero_pairs(t)
        iszero(norm(v)) && delete!(t, k)
    end
    return t
end

function droptol!(t::SparseBlockTensorMap, tol=eps(real(scalartype(t)))^(3 / 4))
    for (k, v) in nonzero_pairs(t)
        norm(v) < tol && delete!(t, k)
    end
end
# Utility
# -------
function Base.copy(t::SparseBlockTensorMap{TT}) where {TT}
    return SparseBlockTensorMap{TT}(deepcopy(t.data), t.codom, t.dom)
end
function Base.delete!(t::SparseBlockTensorMap{TT}, I::CartesianIndex) where {TT}
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
function Base.size(A::SparseTensorArray)
    return (length.(codomain(A.space))..., length.(domain(A.space))...)
end

function Base.getindex(
    A::SparseTensorArray{S,N₁,N₂,T,N}, I::Vararg{Int,N}
) where {S,N₁,N₂,T,N}
    @boundscheck checkbounds(A, I...)
    return get(A.data, CartesianIndex(I)) do
        return fill!(similar(T, eachspace(A)[I...]), zero(scalartype(T)))
    end
end
function Base.setindex!(
    A::SparseTensorArray{S,N₁,N₂,T,N}, v, I::Vararg{Int,N}
) where {S,N₁,N₂,T,N}
    @boundscheck begin
        checkbounds(A, I...)
        checkspaces(A, v, I...)
    end
    A.data[CartesianIndex(I)] = v # implicit converter
    return A
end

function Base.delete!(A::SparseTensorArray, I::Vararg{Int,N}) where {N}
    return delete!(A.data, CartesianIndex(I))
end
Base.delete!(A::SparseTensorArray, I::CartesianIndex) = delete!(A.data, I)

function Base.similar(
    ::SparseTensorArray, ::Type{T}, spaces::TensorMapSumSpace{S,N₁,N₂}
) where {S,N₁,N₂,T<:AbstractTensorMap{<:Any,S,N₁,N₂}}
    N = N₁ + N₂
    return SparseTensorArray{S,N₁,N₂,T,N}(Dict{CartesianIndex{N},T}(), spaces)
end

function Base.copyto!(
    t::SparseTensorArray, v::SubArray{T,N,A}
) where {T,N,A<:SparseTensorArray}
    for (i, j) in zip(eachindex(t), collect(eachindex(parent(v)))[v.indices...])
        if j ∈ nonzero_keys(parent(v))
            t[i] = parent(v)[j]
        end
    end
    return t
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
