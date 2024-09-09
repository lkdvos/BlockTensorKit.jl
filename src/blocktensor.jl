"""
    struct BlockTensorMap{TT<:AbstractTensorMap{E,S,N₁,N₂}} <: AbstractTensorMap{E,S,N₁,N₂}


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
function Base.show(io::IO, ::Type{<:BlockTensorMap{TT}}) where {TT}
    return print(io, "BlockTensorMap{", TT, "}")
end

function blocktensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
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
    data = map(Base.Fix1(TT, undef), SumSpaceIndices(codom ← dom))
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
            t = BlockTensorMap{TT}(undef, V)
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
            t = BlockTensorMap{TT}(undef, V)
            Random.$randfun!(rng, t)
            return t
        end

        function Random.$randfun!(rng::Random.AbstractRNG, t::BlockTensorMap)
            foreach(b -> Random.$randfun!(rng, b), parent(t))
            return t
        end
    end
end

# function BlockTensorMap{E,S,N₁,N₂,A}(
#     ::UndefInitializer, V::TensorMapSumSpace{S,N₁,N₂}
# ) where {E,S,N₁,N₂,A<:AbstractArray{<:AbstractTensorMap{E,S,N₁,N₂}}}
#     return BlockTensorMap{E,S,N₁,N₂,A}(undef, codomain(V), domain(V))
# end
# function BlockTensorMap{E,S,N₁,N₂,A}(
#     ::UndefInitializer, V::TensorMapSpace{S,N₁,N₂}
# ) where {E,S,N₁,N₂,A}
#     V_sumspace = convert(TensorMapSumSpace{S,N₁,N₂}, V)
#     return BlockTensorMap{E,S,N₁,N₂,A}(undef, V_sumspace)
# end

# sparse constructor
# function BlockTensorMap{E,S,N1,N2,A}(
#     ::UndefInitializer, codomain::ProductSumSpace{S,N1}, domain::ProductSumSpace{S,N2}
# ) where {E,S,N1,N2,A<:SparseTensorArray{S,N1,N2,<:AbstractTensorMap{E,S,N1,N2},<:Any}}
#     data = A(undef, codomain ← domain)
#     return BlockTensorMap(data, codomain, domain)
# end
#
# function TensorKit.tensormaptype(
#     ::Type{SumSpace{S}}, N₁::Int, N₂::Int, ::Type{T}
# ) where {S,T<:TensorKit.MatOrNumber}
#     TT = TensorKit.tensormaptype(S, N₁, N₂, T)
#     A = Array{TT,N₁ + N₂}
#     return BlockTensorMap{scalartype(TT),S,N₁,N₂,A}
# end
#
# function TensorKit.tensormaptype(
#     ::Type{SumSpace{S}}, N₁::Int, N₂::Int, ::Type{T}
# ) where {S<:IndexSpace,T<:Union{AbstractTensorMap,AbstractArray{<:AbstractTensorMap}}}
#     A = if T <: TensorKit.MatOrNumber
#         TT = TensorKit.tensormaptype(S, N₁, N₂, T)
#         Array{TT,N₁ + N₂}
#     elseif T <: AbstractTensorMap
#         Array{T,N₁ + N₂}
#     elseif T <: AbstractArray{<:AbstractTensorMap{<:Any,S,N₁,N₂}}
#         T
#     else
#         throw(ArgumentError("tensormaptype: invalid type $T"))
#     end
#     return BlockTensorMap{scalartype(T),S,N₁,N₂,A}
# end

# function sparseblocktensormaptype(::Type{S}, N1::Int, N2::Int, ::Type{T}) where {S,T}
#     TT = T <: AbstractTensorMap{<:Any,S,N1,N2} ? T : TensorKit.tensormaptype(S, N1, N2, T)
#     A = SparseTensorArray{S,N1,N2,TT,N1 + N2}
#     return BlockTensorMap{scalartype(TT),S,N1,N2,A}
# end

# blocktype(::BlockTensorMap{E,S,N₁,N₂,A}) where {E,S,N₁,N₂,A} = A
# blocktype(::Type{BlockTensorMap{E,S,N₁,N₂,A}}) where {E,S,N₁,N₂,A} = A

# function _make_dense(t::BlockTensorMap)
#     tdst = tensormaptype(sumspacetype(spacetype(t)), numout(t), numin(t), eltype(t))(
#         undef, space(t)
#     )
#     for I in eachindex(t)
#         tdst[I] = t[I]
#     end
#     return tdst
# end
# function _make_sparse(t::BlockTensorMap; atol=1e-10)
#     tdst = sparseblocktensormaptype(
#         sumspacetype(spacetype(t)), numout(t), numin(t), eltype(t)
#     )(
#         undef, space(t)
#     )
#     for I in eachindex(t)
#         tmp = t[I]
#         if !isapprox(norm(tmp), zero(scalartype(tmp)); atol)
#             tdst[I] = tmp
#         end
#     end
#     return tdst
# end

# const BlockTensor{E,S,N,A} = BlockTensorMap{E,S,N,0,A}
#
# const BlockTrivialTensorMap{E,S,N₁,N₂,A<:AbstractArray{<:TrivialTensorMap{E,S,N₁,N₂}}} = BlockTensorMap{
#     E,S,N₁,N₂,A
# }
#
# const SparseBlockTensorMap{E,S,N₁,N₂,A<:SparseTensorArray} = BlockTensorMap{E,S,N₁,N₂,A}

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
# function BlockTensorMap{E}(
#     ::UndefInitializer, V::TensorMapSumSpace{S,N₁,N₂}
# ) where {E,S,N₁,N₂}
#     TT = tensormaptype(SumSpace{S}, N₁, N₂, E)
#     return TT(undef, codomain(V), domain(V))
# end
# function BlockTensorMap{E}(
#     ::UndefInitializer, codomain::TensorSumSpace{S}, domain::TensorSumSpace{S}
# ) where {E,S}
#     return BlockTensorMap{E}(undef, codomain ← domain)
# end
# function BlockTensor{E}(::UndefInitializer, V::TensorSumSpace{S}) where {E,S}
#     return BlockTensorMap{E}(undef, V, one(V))
# end

# TODO: the following can probably be avoided by a careful definition change in TensorKit

# AbstractArray Interface
# -----------------------
# mostly pass everything through to data
# Base.ndims(t::BlockTensorMap) = ndims(t.data)
# Base.size(t::BlockTensorMap, args...) = size(t.data, args...)
# Base.length(t::BlockTensorMap) = length(parent(t))
# Base.axes(t::BlockTensorMap, args...) = axes(t.data, args...)
# Base.firstindex(t::BlockTensorMap, args...) = firstindex(t.data, args...)
# Base.first(t::BlockTensorMap) = first(parent(t))
# Base.last(t::BlockTensorMap) = last(parent(t))
# Base.lastindex(t::BlockTensorMap, args...) = lastindex(t.data, args...)
#
# # scalar indexing:
# @inline Base.getindex(t::BlockTensorMap, I::Vararg{Int,N}) where {N} =
#     getindex(parent(t), I...)
# @inline Base.getindex(t::BlockTensorMap, I::CartesianIndex{N}) where {N} =
#     getindex(parent(t), I)
#
# # slicing:
# Base.@propagate_inbounds function Base.getindex(t::BlockTensorMap, I...)
#     newdata = getindex(parent(t), I...)
#     Vslice = eachspace(t)[I...]
#     V = space(Vslice)
#     newdata2 = if ndims(newdata) != ndims(t)
#         reshape(newdata, size(Vslice)) # subspaces do not throw away dims of scalar indices
#     else
#         newdata
#     end
#
#     return BlockTensorMap(newdata2, codomain(V), domain(V))
# end
#
# #
# #
# #
# # Base.getindex(t::BlockTensorMap, args...) = getindex(t.data, args...)
# Base.setindex!(t::BlockTensorMap, args...) = (setindex!(t.data, args...); t)
#
# @inline function Base.setindex!(t::BlockTensorMap, v::AbstractTensorMap, I...)
#     # @assert N == numind(t)
#     @boundscheck begin
#         checkbounds(t, I...)
#         checkspaces(t, v, I...)
#     end
#
#     @inbounds parent(t)[I...] = v
#     return t
# end
# @inline function Base.setindex!(t::BlockTensorMap, v::BlockTensorMap, I...)
#     @boundscheck begin
#         checkbounds(t, I...)
#         checkspaces(t, v, I...)
#     end
#
#     if length(v) == 1
#         t[I...] = only(v)
#         return t
#     end
#
#     Rdest = CartesianIndices(t)[I...]
#     Rsrc = CartesianIndices(v)
#     copyto!(t, Rdest, v, Rsrc)
#     return t
# end
#
# Base.delete!(t::BlockTensorMap, I...) = delete!(parent(t), I...)
#
# Base.eachindex(t::BlockTensorMap) = eachindex(t.data)
# Base.eachindex(style::IndexStyle, t::BlockTensorMap) = eachindex(style, t.data)
# Base.CartesianIndices(t::BlockTensorMap) = CartesianIndices(t.data)
# eachspace(t::BlockTensorMap) = SumSpaceIndices(space(t))
#
# Base.eltype(::Type{BlockTensorMap{E,S,N₁,N₂,A}}) where {E,S,N₁,N₂,A} = eltype(A)
Base.eltype(::Type{<:BlockTensorMap{TT}}) where {TT} = TT
#
# Base.fill!(t::BlockTensorMap, value::Number) = (fill!.(parent(t), value); t)
# Base.only(t::BlockTensorMap) = only(parent(t))
# Base.isempty(t::BlockTensorMap) = isempty(parent(t))
# Base.fill!(t::BlockTensorMap, args...) = (fill!(t.data, args...); t)

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

# Base.delete!(t::BlockTensorMap, I::CartesianIndex) = delete!(t.data, I)

@inline function Base.get(t::BlockTensorMap, key, default)
    @boundscheck checkbounds(t, key)
    return get(t.data, key, default)
end
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
# @inline function Base.setindex!(t::BlockTensorMap, v, I::Vararg{Int,N}) where {N}
#     @assert N == numind(t)
#     @boundscheck begin
#         checkbounds(t, I)
#         checkspaces(t, v, I)
#     end
#
#     @inbounds parent(t)[I...] = v
#     return t
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

# actual implementation
# function Base.similar(
#     ::SparseBlockTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S,N₁,N₂}
# ) where {TorA<:TensorKit.MatOrNumber,S,N₁,N₂}
#     TT = sparseblocktensormaptype(S, N₁, N₂, TorA)
#     return TT(undef, P)
# end
# function Base.similar(
#     ::BlockTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S,N₁,N₂}
# ) where {TorA<:TensorKit.MatOrNumber,S,N₁,N₂}
#     TT = tensormaptype(sumspacetype(S), N₁, N₂, TorA)
#     return TT(undef, P)
# end
# function Base.similar(::AbstractTensorMap, ::Type{TorA},
#                       P::TensorMapSpace{S}) where {TorA,S}
#     N₁ = length(codomain(P))
#     N₂ = length(domain(P))
#     TT = tensormaptype(S, N₁, N₂, TorA)
#     return TT(undef, codomain(P), domain(P))
# end
#

# Show
# ----

function Base.summary(io::IO, x::BlockTensorMap)
    print(io, Base.dims2string(size(x)), " BlockTensorMap(", space(x), ")")
    return nothing
end
# function Base.summary(io::IO, x::SparseBlockTensorMap)
#     print(io, Base.dims2string(size(x)), " SparseBlockTensorMap(", space(x), ")")
#     return nothing
# end

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

function Base.promote_rule(
    ::Type{BlockTensorMap{E₁,S,N₁,N₂,N}}, ::Type{BlockTensorMap{E₂,S,N₁,N₂,N}}
) where {E₁,E₂,S,N₁,N₂,N}
    return BlockTensorMap{promote_type(E₁, E₂),S,N₁,N₂,N}
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

function Base.convert(::Type{TT}, t::BlockTensorMap) where {TT<:TensorMap}
    cod = ProductSpace(join.(codomain(t).spaces))
    dom = ProductSpace(join.(domain(t).spaces))
    tdst = TensorMap{scalartype(t)}(undef, cod ← dom)
    for (c, b) in TK.blocks(t)
        TK.block(tdst, c) .= b
    end
    return convert(TT, tdst)
end
function Base.convert(::Type{TT}, t::BlockTensorMap) where {TT<:BlockTensorMap}
    t isa TT && return t

    tdst = TT(undef, space(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = v
    end
    return tdst
end

function Base.convert(
    ::Type{<:AbstractTensorMap{E,S,N₁,N₂}}, t::BlockTensorMap{E,S,N₁,N₂}
) where {E,S,N₁,N₂}
    cod = ProductSpace(join.(codomain(t).spaces))
    dom = ProductSpace(join.(domain(t).spaces))
    tdst = TensorMap{scalartype(t)}(undef, cod ← dom)
    for (c, b) in TK.blocks(t)
        TK.block(tdst, c) .= b
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
