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

# # getindex and setindex! using Vararg{Int,N} signature is needed for the AbstractArray
# # interface, manually dispatch through to CartesianIndex{N} signature to work with Dict.

Base.delete!(t::BlockTensorMap, I...) = (zerovector!(getindex(t, I...)); t)

# Base.delete!(t::BlockTensorMap, I::CartesianIndex) = delete!(t.data, I)

# @inline function Base.get!(t::BlockTensorMap, I::CartesianIndex)
#     @boundscheck checkbounds(t, I)
#     return get!(t.data, I) do
#         return TensorMap(zeros, scalartype(t), getsubspace(space(t), I))
#     end
# end

# Show
# ----

function Base.summary(io::IO, x::BlockTensorMap)
    print(io, Base.dims2string(size(x)), " BlockTensorMap(", space(x), ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", x::BlockTensorMap)
    summary(io, x)
    isempty(x) && return nothing
    print(io, ":")

    # compute new context
    if !haskey(io, :compact) && length(axes(x, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    if get(io, :limit, false)::Bool && displaysize(io)[1] - 4 <= 0
        return print(io, " …")
    else
        println(io)
    end
    io = IOContext(io, :typeinfo => eltype(x))

    return Base.print_array(io, parent(x))
end

# function Base.show(io::IO, ::MIME"text/plain", x::SparseBlockTensorMap)
#     compact = get(io, :compact, false)::Bool
#     nnz = nonzero_length(x)
#     print(io, Base.join(size(x), "×"), " BlockTensorMap(", space(x), ")")
#     if !compact && nnz != 0
#         println(io, " with ", nnz, " stored entr", nnz == 1 ? "y" : "ies", ":")
#         show_braille(io, x)
#     end
#     return nothing
# end
# function Base.show(io::IO, x::SparseBlockTensorMap)
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

function SparseBlockTensorMap(t::BlockTensorMap)
    tdst = SparseBlockTensorMap{eltype(t)}(undef, codomain(t), domain(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = v
    end
    return tdst
end

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
