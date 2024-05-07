struct SparseTensorArray{S,N₁,N₂,T<:AbstractTensorMap{<:Any,S,N₁,N₂},N} <:
       AbstractArray{T,N}
    data::Dict{CartesianIndex{N},T}
    space::HomSpace{SumSpace{S},N₁,N₂}
    function SparseTensorArray{S,N₁,N₂,T,N}(data::Dict{CartesianIndex{N},T},
                                            space::HomSpace{SumSpace{S},N₁,N₂}) where {S,N₁,
                                                                                       N₂,T,
                                                                                       N}
        N₁ + N₂ == N ||
            throw(TypeError(:SparseTensorArray, SparseTensorArray{S,N₁,N₂,T,N₁ + N₂},
                            SparseTensorArray{S,N₁,N₂,T,N}))
        return new{S,N₁,N₂,T,N}(data, space)
    end
end

# AbstractArray interface
# -----------------------
function Base.size(A::SparseTensorArray)
    return (length.(codomain(A.space))..., length.(domain(A.space))...)
end

function Base.getindex(A::SparseTensorArray{S,N₁,N₂,T,N},
                       I::Vararg{Int,N}) where {S,N₁,N₂,T,N}
    @boundscheck checkbounds(A, I...)
    return get(t.data, CartesianIndex(I)) do
        return similar(T, getsubspace(A.space, CartesianIndex(I)))
    end
end
function Base.setindex!(A::SparseTensorArray{S,N₁,N₂,T,N}, v,
                        I::Vararg{Int,N}) where {S,N₁,N₂,T,N}
    @boundscheck begin
        checkbounds(A, I...)
        checkspaces(t, v, I)
    end
    t.data[I] = v # implicit converter
    return t
end

function Base.similar(::SparseTensorArray, ::Type{T},
                      spaces::HomSpace{SumSpace{S},N₁,N₂}) where {S,N₁,N₂,
                                                                  T<:AbstractTensorMap{<:Any,
                                                                                       S,N₁,
                                                                                       N₂}}
    N = N₁ + N₂
    return SparseTensorArray{S,N₁,N₂,T,N}(Dict{CartesianIndex{N},T}(), spaces)
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

function Base._unsafe_getindex(::IndexCartesian, t::SparseTensorArray{S,N₁,N₂,T,N},
                               I::Vararg{Union{Real,AbstractArray},N}) where {S,N₁,N₂,T,N}
    dest = similar(t, getsubspace(space(t), I...)) # hook into similar to have proper space
    indices = Base.to_indices(t, I)
    for (k, v) in t.data
        newI = _newindices(k.I, indices)
        if newI !== nothing
            dest[newI...] = v
        end
    end
    return dest
end
