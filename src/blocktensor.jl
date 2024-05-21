struct BlockTensorMap{E,S,N₁,N₂,A<:AbstractArray{<:AbstractTensorMap{E,S,N₁,N₂}}} <:
       AbstractTensorMap{E,S,N₁,N₂}
    data::A
    codom::ProductSumSpace{S,N₁}
    dom::ProductSumSpace{S,N₂}

    function BlockTensorMap{E,S,N1,N2,A}(::UndefInitializer,
                                         V::HomSpace{SumSpace{S},
                                                     ProductSpace{SumSpace{S},N1},
                                                     ProductSpace{SumSpace{S},N2}}) where {E,
                                                                                           S,
                                                                                           N1,
                                                                                           N2,
                                                                                           A<:AbstractArray}
        sz = map(length, (codomain(V).spaces..., domain(V).spaces...))
        data = A(undef, sz)
        return new{E,S,N1,N2,A}(data, codomain(V), domain(V))
    end
    # constructor enforces N = N₁ + N₂
    function BlockTensorMap(data::AbstractArray{<:AbstractTensorMap{E,S,N₁,N₂}},
                            codom::ProductSumSpace{S,N₁},
                            dom::ProductSumSpace{S,N₂}) where {E,S,N₁,N₂}
        N₁ + N₂ == ndims(data) || throw(TypeError(:BlockTensorMap,
                                                  BlockTensorMap{E,S,N₁,N₂,AbstractArray{E,N₁ + N₂}},
                                                  BlockTensorMap{E,S,N₁,N₂,typeof(data)}))
        return new{E,S,N₁,N₂,typeof(data)}(data, codom, dom)
    end

    # function BlockTensorMap{E,S,N₁,N₂}(data::Dict{CartesianIndex{N},AbstractTensorMap{E,S,N₁,N₂}},
    #                                    codom::ProductSumSpace{S,N₁},
    #                                    dom::ProductSumSpace{S,N₂}) where {
    #                                    E,S,N₁,N₂,N}
    #     N₁ + N₂ == N ||
    #         throw(TypeError(:BlockTensorMap, BlockTensorMap{E,S,N₁,N₂,N₁+N₂},
    #                         BlockTensorMap{E,S,N₁,N₂,N}))
    #     return new{E,S,N₁,N₂,N}(data, codom, dom)
    # end
end

# alias for switching parameters
# const BlockTensorArray{E,N} = BlockTensorMap{E,S,N₁,N₂,N} where {S,N₁,N₂}
# const BlockTensorArray{N} = BlockTensorMap{E,S,N₁,N₂,N} where {E,S,N₁,N₂} # TODO: consider this alias

# default type parameters
# function BlockTensorMap{E,S,N₁,N₂}(args...) where {E,S,N₁,N₂}
#     return BlockTensorMap{E,S,N₁,N₂,N₁ + N₂}(args...)
# end
# function BlockTensorMap{E,S,N₁,N₂,N}(::UndefInitializer,
#                                      V::TensorMapSumSpace{S,N₁,N₂}) where {E,S,N₁,N₂,N}
#     return BlockTensorMap{E,S,N₁,N₂,N}(undef, codomain(V), domain(V))
# end

# Constructors (Used to have a tensortype parameter, changed it to scalartype parameter)
# ------------
function BlockTensorMap{E}(::UndefInitializer, codom::ProductSumSpace{S,N₁},
                           dom::ProductSumSpace{S,N₂}) where {E<:Number,S,N₁,N₂}
    data = Array{tensormaptype(S, N₁, N₂, E)}(undef, length.(codom.spaces)...,
                                              length.(dom.spaces)...)
    return BlockTensorMap(data, codom, dom)
end

function BlockTensorMap{E}(::UndefInitializer, P::TensorMapSumSpace) where {E}
    return BlockTensorMap{E}(undef, codomain(P), domain(P))
end

for f in (:zeros, :ones, :randn, :rand)
    @eval function Base.$f(::Type{E}, V::TensorMapSumSpace) where {E}
        t = BlockTensorMap{E}(undef, V)
        for (I, S) in enumerate(eachspace(t))
            t[I] = TensorMap($f, E, S)
        end
        return t
    end
end
for f in (:randn, :rand)
    f! = Symbol(f, :!)
    @eval function Random.$f!(t::BlockTensorMap)
        for I in eachindex(t)
            tmp = t[I]
            for (_, b) in TensorKit.blocks(tmp)
                Random.$f!(b)
            end
            t[I] = tmp
        end
        return t
    end
end

# AbstractArray Interface
# -----------------------
# mostly pass everything through to data
Base.size(t::BlockTensorMap, args...) = size(t.data, args...)
Base.axes(t::BlockTensorMap, args...) = axes(t.data, args...)

# TODO: do we want slicing?
Base.getindex(t::BlockTensorMap, args...) = getindex(t.data, args...)
Base.setindex!(t::BlockTensorMap, args...) = (setindex!(t.data, args...); t)

Base.eachindex(t::BlockTensorMap) = eachindex(t.data)
Base.eachindex(style::IndexStyle, t::BlockTensorMap) = eachindex(style, t.data)
eachspace(t::BlockTensorMap) = SumSpaceIndices(space(t))

Base.eltype(::Type{BlockTensorMap{E,S,N₁,N₂,A}}) where {E,S,N₁,N₂,A} = eltype(A)

Base.fill!(t::BlockTensorMap, args...) = (fill!(t.data, args...); t)

Base.parent(t::BlockTensorMap) = t.data

# Utility
# -------

TK.storagetype(::Type{BlockTensorMap{E,S,N1,N2,A}}) where {E,S,N1,N2,A} = A

# # getindex and setindex! using Vararg{Int,N} signature is needed for the AbstractArray
# # interface, manually dispatch through to CartesianIndex{N} signature to work with Dict.

# Base.delete!(t::BlockTensorMap, I::CartesianIndex) = delete!(t.data, I)

# @inline function Base.get(t::BlockTensorMap, I::CartesianIndex)
#     @boundscheck checkbounds(t, I)
#     return get(t.data, I) do
#         return TensorMap(zeros, scalartype(t), getsubspace(space(t), I))
#     end
# end

# @inline function Base.get!(t::BlockTensorMap, I::CartesianIndex)
#     @boundscheck checkbounds(t, I)
#     return get!(t.data, I) do
#         return TensorMap(zeros, scalartype(t), getsubspace(space(t), I))
#     end
# end

# @propagate_inbounds function Base.getindex(t::BlockTensorArray{E,N},
#                                            I::Vararg{Int,N}) where {E,N}
#     return getindex(t, CartesianIndex(I))
# end
# @inline function Base.getindex(t::BlockTensorArray{E,N},
#                                I::CartesianIndex{N}) where {E,N}
#     return get(t, I)
# end

# @propagate_inbounds function Base.setindex!(t::BlockTensorArray{E,N}, v,
#                                             I::Vararg{Int,N}) where {E,N}
#     return setindex!(t, v, CartesianIndex(I))
# end
# @propagate_inbounds function Base.setindex!(t::BlockTensorArray{E₁,N},
#                                             v::BlockTensorArray{E₂,N},
#                                             I::CartesianIndex{N}) where {E₁,E₂,N}
#     return setindex!(t, only(v), I)
# end
# @inline function Base.setindex!(t::BlockTensorArray{E,N},
#                                 v,
#                                 I::CartesianIndex{N}) where {E,N}
#     @boundscheck begin
#         checkbounds(t, I)
#         checkspaces(t, v, I)
#     end
#     # TODO: consider if it's worth it to check if v is zero
#     t.data[I] = v
#     return t
# end

# non-scalar indexing
# -------------------
# specialisations to have scalar indexing return a TensorMap
# while non-scalar indexing yields a BlockTensorMap

# _newindex(i::Int, range::Int) = i == range ? (1,) : nothing
# function _newindex(i::Int, range::AbstractVector{Int})
#     k = findfirst(==(i), range)
#     return k === nothing ? nothing : (k,)
# end
# _newindices(::Tuple{}, ::Tuple{}) = ()
# function _newindices(I::Tuple, indices::Tuple)
#     i = _newindex(I[1], indices[1])
#     Itail = _newindices(Base.tail(I), Base.tail(indices))
#     (i === nothing || Itail === nothing) && return nothing
#     return (i..., Itail...)
# end

# function Base._unsafe_getindex(::IndexCartesian, t::BlockTensorArray{T,N},
#                                I::Vararg{Union{Real,AbstractArray},N}) where {T,N}
#     dest = similar(t, getsubspace(space(t), I...)) # hook into similar to have proper space
#     indices = Base.to_indices(t, I)
#     shape = length.(Base.index_shape(indices...))
#     # size(dest) == shape || Base.throw_checksize_error(dest, shape)
#     for (k, v) in nonzero_pairs(t)
#         newI = _newindices(k.I, indices)
#         if newI !== nothing
#             dest[newI...] = v
#         end
#     end
#     return dest
# end

# Base.similar
# ------------
# specialisations to have `similar` behave with spaces, and disallow undefined options.

# 4 arguments
function Base.similar(t::BlockTensorMap, E::Type, codomain::VectorSpace,
                      domain::VectorSpace)
    return similar(t, E, codomain ← domain)
end
# 3 arguments
function Base.similar(t::BlockTensorMap, codomain::VectorSpace, domain::VectorSpace)
    return similar(t, scalartype(t), codomain ← domain)
end
function Base.similar(t::BlockTensorMap, E::Type, codomain::VectorSpace)
    return similar(t, E, codomain ← one(codomain))
end
# 2 arguments
function Base.similar(t::BlockTensorMap, codomain::VectorSpace)
    return similar(t, scalartype(t), codomain ← one(codomain))
end
Base.similar(t::BlockTensorMap, P::TensorMapSpace) = similar(t, scalartype(t), P)
Base.similar(t::BlockTensorMap, E::Type) = similar(t, E, space(t))
# 1 argument
Base.similar(t::BlockTensorMap) = similar(t, scalartype(t), space(t))

# actual implementation
function Base.similar(t::BlockTensorMap, ::Type{E},
                      P::TensorMapSumSpace{S,N₁,N₂}) where {E,S,N₁,N₂}
    T = tensormaptype(S, N₁, N₂, E)
    A = TK.similarstoragetype(t, T)
    return BlockTensorMap{E,S,N₁,N₂,A}(undef, P)
end

# Space checking
# --------------

function checkspaces(t::BlockTensorMap{E,S,N₁,N₂,N}, v::AbstractTensorMap{E,S,N₁,N₂},
                     I::CartesianIndex{N}) where {E,S,N₁,N₂,N}
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
# function Base.show(io::IO, ::MIME"text/plain", x::BlockTensorMap)

# function Base.show(io::IO, ::MIME"text/plain", x::BlockTensorMap)
#     compact = get(io, :compact, false)::Bool
#     nnz = nonzero_length(x)
#     print(io, Base.join(size(x), "×"), " BlockTensorMap(", space(x), ")")
#     if !compact && nnz != 0
#         println(io, " with ", nnz, " stored entr", nnz == 1 ? "y" : "ies", ":")
#         show_braille(io, x)
#     end
#     return nothing
# end
# function Base.show(io::IO, x::BlockTensorMap)
#     compact = get(io, :compact, false)::Bool
#     nnz = nonzero_length(x)
#     print(io, Base.join(size(x), "×"), " BlockTensorMap(", space(x), ")")
#     if !compact && nnz != 0
#         println(io, " with ", nnz, " stored entr", nnz == 1 ? "y" : "ies", ":")
#         show_elements(io, x)
#     end
#     return nothing
# end

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

function Base.promote_rule(::Type{BlockTensorMap{E₁,S,N₁,N₂,N}},
                           ::Type{BlockTensorMap{E₂,S,N₁,N₂,N}}) where {E₁,E₂,S,N₁,N₂,N}
    return BlockTensorMap{promote_type(E₁, E₂),S,N₁,N₂,N}
end

function Base.convert(::Type{BlockTensorMap{E₁,S,N₁,N₂,N}},
                      t::BlockTensorMap{E₂,S,N₁,N₂,N}) where {E₁,E₂,S,N₁,N₂,N}
    E₁ === E₂ && return t
    tdst = BlockTensorMap{E₁,S,N₁,N₂,N}(undef, codomain(t), domain(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = add!(zerovector(v, E₁), v, One())
    end
    return tdst
end

function Base.convert(::Type{BlockTensorMap},
                      t::AbstractTensorMap{E,S,N₁,N₂}) where {E,S,N₁,N₂}
    tdst = BlockTensorMap{E,S,N₁,N₂}(undef,
                                     convert(ProductSumSpace{S,N₁}, codomain(t)),
                                     convert(ProductSumSpace{S,N₂}, domain(t)))
    tdst[1] = t
    return tdst
end

function Base.convert(::Type{<:AbstractTensorMap{E,S,N₁,N₂}},
                      t::BlockTensorMap{E,S,N₁,N₂}) where {E,S,N₁,N₂}
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
function Base.copy(tsrc::BlockTensorMap{E,S,N1,N2,N}) where {E,S,N1,N2,N}
    tdst = similar(tsrc)
    for (key, value) in tsrc.data
        tdst[key] = copy(value)
    end
    return tdst
end

Base.haskey(t::BlockTensorMap, I::CartesianIndex) = haskey(t.data, I)
function Base.haskey(t::BlockTensorMap, i::Int)
    return haskey(t.data, CartesianIndices(t)[i])
end
