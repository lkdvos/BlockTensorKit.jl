#! format: off
struct BlockTensorMap{S<:IndexSpace,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂},N} <:
       AbstractArray{T,N}
    data::Dict{CartesianIndex{N},T}
    codom::ProductSumSpace{S,N₁}
    dom::ProductSumSpace{S,N₂}
    function BlockTensorMap{S,N₁,N₂,T,N}(::UndefInitializer, codom::ProductSumSpace{S,N₁},
                                       dom::ProductSumSpace{S,N₂}) where {
                                       S,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂},N}
        N₁ + N₂ == N ||
            throw(TypeError(:BlockTensorMap, BlockTensorMap{S,N₁,N₂,T,N₁+N₂},
                            BlockTensorMap{S,N₁,N₂,T,N}))
        return new{S,N₁,N₂,T,N}(Dict{CartesianIndex{N},T}(), codom, dom)
    end
    function BlockTensorMap{S,N₁,N₂,T}(data::Dict{CartesianIndex{N},T},
                                       codom::ProductSumSpace{S,N₁},
                                       dom::ProductSumSpace{S,N₂}) where {
                                       S,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂},N}
        N₁ + N₂ == N ||
            throw(TypeError(:BlockTensorMap, BlockTensorMap{S,N₁,N₂,T,N₁+N₂},
                            BlockTensorMap{S,N₁,N₂,T,N}))
        return new{S,N₁,N₂,T,N}(data, codom, dom)
    end
end
#! format: on

# alias for switching parameters
const BlockTensorArray{T,N} = BlockTensorMap{S,N₁,N₂,T,N} where {S,N₁,N₂}

# default type parameters
function BlockTensorMap{S,N₁,N₂,T}(args...) where {S,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂}}
    return BlockTensorMap{S,N₁,N₂,T,N₁ + N₂}(args...)
end
function BlockTensorMap{S,N₁,N₂,T,N}(::UndefInitializer,
                                     V::TensorMapSumSpace{S,N₁,N₂}) where {S,N₁,N₂,T,N}
    return BlockTensorMap{S,N₁,N₂,T,N}(undef, codomain(V), domain(V))
end

# Constructors
# ------------
function BlockTensorMap(::UndefInitializer, ::Type{T}, codom::ProductSumSpace{S,N₁},
                        dom::ProductSumSpace{S,N₂}) where {T,S,N₁,N₂}
    T′ = T <: AbstractTensorMap{S,N₁,N₂} ? T : tensormaptype(S, N₁, N₂, T)
    return BlockTensorMap{S,N₁,N₂,T′}(undef, codom, dom)
end
function BlockTensorMap(::UndefInitializer, T::Type, P::TensorMapSumSpace)
    return BlockTensorMap(undef, T, codomain(P), domain(P))
end

# AbstractArray Interface
# -----------------------
Base.size(t::BlockTensorMap) = (length.(t.codom.spaces)..., length.(t.dom.spaces)...)
function Base.size(t::BlockTensorMap{S,N₁}, i::Int) where {S,N₁}
    return i > N₁ ? length(t.dom.spaces[i - N₁]) : length(t.codom.spaces[i])
end

# getindex and setindex! using Vararg{Int,N} signature is needed for the AbstractArray
# interface, manually dispatch through to CartesianIndex{N} signature to work with Dict.

Base.delete!(t::BlockTensorMap, I::CartesianIndex) = delete!(t.data, I)

@inline function Base.get(t::BlockTensorMap, I::CartesianIndex)
    @boundscheck checkbounds(t, I)
    return get(t.data, I) do
        return TensorMap(zeros, scalartype(t), getsubspace(space(t), I))
    end
end

@inline function Base.get!(t::BlockTensorMap, I::CartesianIndex)
    @boundscheck checkbounds(t, I)
    return get!(t.data, I) do
        return TensorMap(zeros, scalartype(t), getsubspace(space(t), I))
    end
end

@propagate_inbounds function Base.getindex(t::BlockTensorArray{T,N},
                                           I::Vararg{Int,N}) where {T,N}
    return getindex(t, CartesianIndex(I))
end
@inline function Base.getindex(t::BlockTensorArray{T,N},
                               I::CartesianIndex{N}) where {T,N}
    return get(t, I)
end

@propagate_inbounds function Base.setindex!(t::BlockTensorArray{T,N}, v,
                                            I::Vararg{Int,N}) where {T,N}
    return setindex!(t, v, CartesianIndex(I))
end
@propagate_inbounds function Base.setindex!(t::BlockTensorArray{T₁,N},
                                            v::BlockTensorArray{T₂,N},
                                            I::CartesianIndex{N}) where {T₁,T₂,N}
    return setindex!(t, only(v), I)
end
@inline function Base.setindex!(t::BlockTensorArray{T,N},
                                v,
                                I::CartesianIndex{N}) where {T,N}
    @boundscheck begin
        checkbounds(t, I)
        checkspaces(t, v, I)
    end
    # TODO: consider if it's worth it to check if v is zero
    t.data[I] = v
    return t
end

# non-scalar indexing
# -------------------
# specialisations to have scalar indexing return a TensorMap
# while non-scalar indexing yields a BlockTensorMap

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

function Base._unsafe_getindex(::IndexCartesian, t::BlockTensorArray{T,N},
                               I::Vararg{Union{Real,AbstractArray},N}) where {T,N}
    dest = similar(t, getsubspace(space(t), I...)) # hook into similar to have proper space
    indices = Base.to_indices(t, I)
    shape = length.(Base.index_shape(indices...))
    # size(dest) == shape || Base.throw_checksize_error(dest, shape)
    for (k, v) in nonzero_pairs(t)
        newI = _newindices(k.I, indices)
        if newI !== nothing
            dest[newI...] = v
        end
    end
    return dest
end

# Base.similar
# ------------
# specialisations to have `similar` behave with spaces, and disallow undefined options.

# 4 arguments
function Base.similar(t::BlockTensorMap, T::Type, codomain::VectorSpace,
                      domain::VectorSpace)
    return similar(t, T, codomain ← domain)
end
# 3 arguments
function Base.similar(t::BlockTensorMap, codomain::VectorSpace, domain::VectorSpace)
    return similar(t, scalartype(t), codomain ← domain)
end
function Base.similar(t::BlockTensorMap, T::Type, codomain::VectorSpace)
    return similar(t, T, codomain ← one(codomain))
end
# 2 arguments
function Base.similar(t::BlockTensorMap, codomain::VectorSpace)
    return similar(t, scalartype(t), codomain ← one(codomain))
end
Base.similar(t::BlockTensorMap, P::TensorMapSpace) = similar(t, scalartype(t), P)
Base.similar(t::BlockTensorMap, T::Type) = similar(t, T, space(t))
# 1 argument
Base.similar(t::BlockTensorMap) = similar(t, scalartype(t), space(t))

# actual implementation
function Base.similar(::BlockTensorMap, T::Type, P::TensorMapSumSpace)
    return BlockTensorMap(undef, T, P)
end

# Space checking
# --------------

function checkspaces(t::BlockTensorMap{S,N₁,N₂,T,N}, v::AbstractTensorMap{S,N₁,N₂},
                     I::CartesianIndex{N}) where {S,N₁,N₂,T,N}
    getsubspace(space(t), I) == space(v) ||
        throw(SpaceMismatch("inserting a tensor of space $(space(v)) at index $I into a tensor of space $(getsubspace(space(t), I))"))
    return nothing
end
function checkspaces(t::BlockTensorMap)
    for (I, v) in nonzero_pairs(t)
        checkspaces(t, v, I)
    end
    return nothing
end

# Data iterators
# --------------
# (stolen from SparseArrayKit)

nonzero_pairs(a::BlockTensorMap) = pairs(a.data)
nonzero_keys(a::BlockTensorMap) = keys(a.data)
nonzero_values(a::BlockTensorMap) = values(a.data)
nonzero_length(a::BlockTensorMap) = length(a.data)

# Show
# ----
function Base.show(io::IO, ::MIME"text/plain", x::BlockTensorMap)
    compact = get(io, :compact, false)::Bool
    nnz = nonzero_length(x)
    print(io, Base.join(size(x), "×"), " BlockTensorMap(", space(x), ")")
    if !compact && nnz != 0
        println(io, " with ", nnz, " stored entr", nnz == 1 ? "y" : "ies", ":")
        show_braille(io, x)
    end
    return nothing
end
function Base.show(io::IO, x::BlockTensorMap)
    compact = get(io, :compact, false)::Bool
    nnz = nonzero_length(x)
    print(io, Base.join(size(x), "×"), " BlockTensorMap(", space(x), ")")
    if !compact && nnz != 0
        println(io, " with ", nnz, " stored entr", nnz == 1 ? "y" : "ies", ":")
        show_elements(io, x)
    end
    return nothing
end

function show_elements(io::IO, x::BlockTensorMap)
    nzind = nonzero_keys(x)
    length(nzind) == 0 && return nothing
    limit = get(io, :limit, false)::Bool
    compact = get(io, :compact, true)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    pads = map(1:ndims(x)) do i
        return ndigits(maximum(getindex.(nzind, i)))
    end
    io = IOContext(io, :compact => compact)
    nz_pairs = sort(collect(nonzero_pairs(x)); by=first)
    for (k, (ind, val)) in enumerate(nz_pairs)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            println(io, "  ", '[', Base.join(lpad.(Tuple(ind), pads), ","), "]  =  ", val)
        elseif k == half_screen_rows
            println(io, "   ", Base.join(" " .^ pads, " "), "   \u22ee")
        end
    end
end

# adapted from SparseArrays.jl
const brailleBlocks = UInt16['⠁', '⠂', '⠄', '⡀', '⠈', '⠐', '⠠', '⢀']
function show_braille(io::IO, x::BlockTensorMap)
    m = prod(getindices(size(x), codomainind(x)))
    n = prod(getindices(size(x), domainind(x)))
    reshape_helper = reshape(CartesianIndices((m, n)), size(x))

    # The maximal number of characters we allow to display the matrix
    local maxHeight::Int, maxWidth::Int
    maxHeight = displaysize(io)[1] - 4 # -4 from [Prompt, header, newline after elements, new prompt]
    maxWidth = displaysize(io)[2] ÷ 2

    if get(io, :limit, true) && (m > 4maxHeight || n > 2maxWidth)
        s = min(2maxWidth / n, 4maxHeight / m)
        scaleHeight = floor(Int, s * m)
        scaleWidth = floor(Int, s * n)
    else
        scaleHeight = m
        scaleWidth = n
    end

    # Make sure that the matrix size is big enough to be able to display all
    # the corner border characters
    if scaleHeight < 8
        scaleHeight = 8
    end
    if scaleWidth < 4
        scaleWidth = 4
    end

    brailleGrid = fill(UInt16(10240), (scaleWidth - 1) ÷ 2 + 4, (scaleHeight - 1) ÷ 4 + 1)
    brailleGrid[1, :] .= '⎢'
    brailleGrid[end - 1, :] .= '⎥'
    brailleGrid[1, 1] = '⎡'
    brailleGrid[1, end] = '⎣'
    brailleGrid[end - 1, 1] = '⎤'
    brailleGrid[end - 1, end] = '⎦'
    brailleGrid[end, :] .= '\n'

    rowscale = max(1, scaleHeight - 1) / max(1, m - 1)
    colscale = max(1, scaleWidth - 1) / max(1, n - 1)

    for I′ in nonzero_keys(x)
        I = reshape_helper[I′]
        si = round(Int, (I[1] - 1) * rowscale + 1)
        sj = round(Int, (I[2] - 1) * colscale + 1)

        k = (sj - 1) ÷ 2 + 2
        l = (si - 1) ÷ 4 + 1
        p = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)

        brailleGrid[k, l] |= brailleBlocks[p]
    end

    foreach(c -> print(io, Char(c)), @view brailleGrid[1:(end - 1)])
    return nothing
end

# Converters
# ----------

function Base.promote_rule(::Type{BlockTensorMap{S,N₁,N₂,T₁,N}},
                           ::Type{BlockTensorMap{S,N₁,N₂,T₂,N}}) where {S,N₁,N₂,T₁,T₂,N}
    return BlockTensorMap{S,N₁,N₂,promote_type(T₁, T₂),N}
end

function Base.convert(::Type{BlockTensorMap{S,N₁,N₂,T₁,N}},
                      t::BlockTensorMap{S,N₁,N₂,T₂,N}) where {S,N₁,N₂,T₁,T₂,N}
    T₁ === T₂ && return t
    tdst = BlockTensorMap{S,N₁,N₂,T₁,N}(undef, codomain(t), domain(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = convert(T₁, v)
    end
    return tdst
end

function Base.convert(::Type{BlockTensorMap}, t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    tdst = BlockTensorMap{S,N₁,N₂,typeof(t)}(undef,
                                             convert(ProductSumSpace{S,N₁}, codomain(t)),
                                             convert(ProductSumSpace{S,N₂}, domain(t)))
    tdst[1] = t
    return tdst
end

function Base.convert(::Type{<:AbstractTensorMap{S,N₁,N₂}},
                      t::BlockTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    cod = ProductSpace(join.(codomain(t).spaces))
    dom = ProductSpace(join.(domain(t).spaces))
    tdst = TensorMap(undef, scalartype(t), cod ← dom)
    for (c, b) in TK.blocks(t)
        TK.block(tdst, c) .= b
    end
    return tdst
end

# Utility
# -------

function Base.copy(t::BlockTensorMap{S,N₁,N₂,T,N}) where {S,N₁,N₂,T,N}
    return BlockTensorMap{S,N₁,N₂,T,N}(copy(t.data), codomain(t), domain(t))
end

Base.haskey(t::BlockTensorMap, I::CartesianIndex) = haskey(t.data, I)
function Base.haskey(t::BlockTensorMap, i::Int)
    return haskey(t.data, CartesianIndices(t)[i])
end
