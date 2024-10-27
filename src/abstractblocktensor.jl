"""
    AbstractBlockTensorMap{E,S,N₁,N₂}

Abstract supertype for tensor maps that have additional block structure, i.e. they act on vector spaces
that have a direct sum structure. These behave like `AbstractTensorMap` but have additional methods to
facilitate indexing and manipulation of the block structure.
"""
abstract type AbstractBlockTensorMap{E,S,N₁,N₂} <: AbstractTensorMap{E,S,N₁,N₂} end

# AbstractArray Interface
# -----------------------
# mostly pass everything through to the parent array, but with additional features for slicing
Base.eltype(t::AbstractBlockTensorMap) = eltype(parent(t))

Base.ndims(t::AbstractBlockTensorMap) = ndims(parent(t))
Base.size(t::AbstractBlockTensorMap, args...) = size(parent(t), args...)
Base.length(t::AbstractBlockTensorMap) = length(parent(t))
Base.axes(t::AbstractBlockTensorMap, args...) = axes(parent(t), args...)

Base.first(t::AbstractBlockTensorMap) = first(parent(t))
Base.last(t::AbstractBlockTensorMap) = last(parent(t))
Base.lastindex(t::AbstractBlockTensorMap, args...) = lastindex(parent(t), args...)
Base.firstindex(t::AbstractBlockTensorMap, args...) = firstindex(parent(t), args...)

Base.CartesianIndices(t::AbstractBlockTensorMap) = CartesianIndices(parent(t))
Base.eachindex(t::AbstractBlockTensorMap) = eachindex(parent(t))

Base.keys(l::Base.IndexStyle, t::AbstractBlockTensorMap) = keys(l, parent(t))
Base.haskey(t::AbstractBlockTensorMap, args...) = haskey(parent(t), args...)

Base.only(t::AbstractBlockTensorMap) = only(parent(t))
Base.isempty(t::AbstractBlockTensorMap) = isempty(parent(t))

# scalar indexing is dispatched through:
@inline Base.getindex(t::AbstractBlockTensorMap, I::Vararg{Int,N}) where {N} =
    getindex(parent(t), I...)
@inline Base.getindex(t::AbstractBlockTensorMap, I::CartesianIndex{N}) where {N} =
    getindex(parent(t), I)

# slicing getindex needs to correctly allocate output blocktensor:
Base.@propagate_inbounds function Base.getindex(t::AbstractBlockTensorMap, I...)
    V = space(eachspace(t)[I...])
    tdst = similar(t, V)
    copyto!(parent(tdst), view(parent(t), I...))
    return tdst
end

# TODO: check if this fallback is fair
@inline Base.setindex!(t::AbstractBlockTensorMap, args...) = (
    setindex!(parent(t), args...); t
)

# setindex verifies structure is correct
@inline function Base.setindex!(t::AbstractBlockTensorMap, v::AbstractTensorMap, I...)
    @boundscheck begin
        checkbounds(t, I...)
        checkspaces(t, v, I...)
    end
    @inbounds parent(t)[I...] = v
    return t
end
# setindex with blocktensor needs to correctly slice-assign
@inline function Base.setindex!(t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, I...)
    @boundscheck begin
        checkbounds(t, I...)
        checkspaces(t, v, I...)
    end

    copyto!(view(parent(t), I...), parent(v))
    return t
end

@inline function Base.get(t::AbstractBlockTensorMap, key, default)
    @boundscheck checkbounds(t, key)
    return get(parent(t), key, default)
end

function Base.copyto!(
    tdst::AbstractBlockTensorMap,
    Rdest::CartesianIndices,
    tsrc::AbstractBlockTensorMap,
    Rsrc::CartesianIndices,
)
    copyto!(parent(tdst), Rdest, parent(tsrc), Rsrc)
    return tdst
end

# generic implementation for AbstractTensorMap with Sumspace -> returns `BlockTensorMap`
# function Base.similar(
#     ::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S}
# ) where {TorA<:TensorKit.MatOrNumber,S}
#     N₁ = length(codomain(P))
#     N₂ = length(domain(P))
#     TT = blocktensormaptype(S, N₁, N₂, TorA)
#     return TT(undef, codomain(P), domain(P))
# end
# disambiguate
# function Base.similar(
#     t::TensorKit.AdjointTensorMap, T::Type{TorA}, P::TensorMapSumSpace{S}
# ) where {TorA<:TensorKit.MatOrNumber,S}
#     @invoke Base.similar(t::TensorKit.AdjointTensorMap, T::Type{TorA}, P::TensorMapSpace)
# end

# make sure tensormap specializations are not used for sumspaces:
function Base.similar(
    t::AbstractTensorMap, ::Type{TorA}, P::TensorMapSumSpace{S}
) where {S,TorA}
    if TorA <: Number
        T = TorA
        A = TensorKit.similarstoragetype(t, T)
    elseif TorA <: DenseVector
        A = TorA
        T = scalartype(A)
    else
        throw(ArgumentError("Type $TorA not supported for similar"))
    end
    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    TT = TensorMap{T,S,N₁,N₂,A}
    return BlockTensorMap{TT}(undef, P)
end

# AbstractTensorMap Interface
# ---------------------------
# TODO: do we really want this:
# note: this goes along with the specializations of Base.similar above...
function TensorKit.tensormaptype(
    ::Type{SumSpace{S}}, N₁::Int, N₂::Int, TorA::Type
) where {S}
    return blocktensormaptype(S, N₁, N₂, TorA)
end

eachspace(t::AbstractBlockTensorMap) = SumSpaceIndices(space(t))

# TODO: delete this method
@inline function Base.getindex(t::AbstractBlockTensorMap, ::Nothing, ::Nothing)
    sectortype(t) === Trivial || throw(SectorMismatch())
    return mortar(map(x -> x[nothing, nothing], parent(t)))
end
@inline function Base.getindex(
    t::AbstractBlockTensorMap{E,S,N₁,N₂}, f₁::FusionTree{I,N₁}, f₂::FusionTree{I,N₂}
) where {E,S,I,N₁,N₂}
    sectortype(S) === I || throw(SectorMismatch())
    return mortar(map(x -> x[f₁, f₂], parent(t)))
end

function TensorKit.block(t::AbstractBlockTensorMap, c::Sector)
    sectortype(t) == typeof(c) || throw(SectorMismatch())

    rows = prod(getindices(size(t), codomainind(t)))
    cols = prod(getindices(size(t), domainind(t)))
    @assert rows != 0 && cols != 0 "to be added"

    allblocks = map(Base.Fix2(block, c), parent(t))
    return mortar(reshape(allblocks, rows, cols))
end

# TODO: this might get fixed once new tensormap is implemented
TensorKit.blocks(t::AbstractBlockTensorMap) = ((c, block(t, c)) for c in blocksectors(t))
TensorKit.blocksectors(t::AbstractBlockTensorMap) = blocksectors(space(t))
TensorKit.hasblock(t::AbstractBlockTensorMap, c::Sector) = c in blocksectors(t)

function TensorKit.storagetype(::Type{TT}) where {TT<:AbstractBlockTensorMap}
    return if isconcretetype(eltype(TT))
        storagetype(eltype(TT))
    else
        Vector{scalartype(TT)}
    end
end

function TensorKit.fusiontrees(t::AbstractBlockTensorMap)
    sectortype(t) === Trivial && return ((nothing, nothing),)
    blocksectoriterator = blocksectors(space(t))
    rowr, _ = TK._buildblockstructure(codomain(t), blocksectoriterator)
    colr, _ = TK._buildblockstructure(domain(t), blocksectoriterator)
    return TK.TensorKeyIterator(rowr, colr)
end

# getindex and setindex checking
# ------------------------------
Base.checkbounds(t::AbstractBlockTensorMap, I...) = checkbounds(parent(t), I...)
function Base.checkbounds(::Type{Bool}, t::AbstractBlockTensorMap, I...)
    return checkbounds(Bool, parent(t), I...)
end

# TODO: make this also have Bool as first argument
function checkspaces(t::AbstractBlockTensorMap, v::AbstractTensorMap, I...)
    space(v) == eachspace(t)[I...] || throw(
        SpaceMismatch(
            "inserting a tensor of space $(space(v)) at index $I into a tensor of space $(eachspace(t)[I...])",
        ),
    )
    return nothing
end
function checkspaces(t::AbstractBlockTensorMap, v::AbstractBlockTensorMap, I...)
    V_slice = eachspace(t)[I...]
    if V_slice isa SumSpaceIndices
        space(v) == space(V_slice) || throw(
            SpaceMismatch(
                "inserting a tensor of space $(space(v)) at index $I into a tensor of space $(space(V_slice))",
            ),
        )
    else
        space(only(v)) == V_slice || throw(
            SpaceMismatch(
                "inserting a tensor of space $(space(only(v))) at index $I into a tensor of space $(V_slice)",
            ),
        )
    end
end
function checkspaces(t::AbstractBlockTensorMap)
    iter = SumSpaceIndices(space(t))
    for I in eachindex(iter)
        iter[I] == space(t[I]) || throw(
            SpaceMismatch(
                "index $I has space $(iter[I]) but tensor has space $(space(t[I]))"
            ),
        )
    end
    return nothing
end

# Conversion
# ----------
function Base.convert(::Type{T}, t::AbstractBlockTensorMap) where {T<:TensorMap}
    cod = ProductSpace{spacetype(t),numout(t)}(join.(codomain(t).spaces))
    dom = ProductSpace{spacetype(t),numin(t)}(join.(domain(t).spaces))

    tdst = similar(t, cod ← dom)
    for (f₁, f₂) in fusiontrees(tdst)
        tdst[f₁, f₂] .= t[f₁, f₂]
    end

    return convert(T, tdst)
end
# disambiguate
function Base.convert(::Type{TensorMap}, t::AbstractBlockTensorMap)
    cod = ProductSpace{spacetype(t),numout(t)}(join.(codomain(t).spaces))
    dom = ProductSpace{spacetype(t),numin(t)}(join.(domain(t).spaces))

    tdst = similar(t, cod ← dom)
    for (f₁, f₂) in fusiontrees(tdst)
        tdst[f₁, f₂] .= t[f₁, f₂]
    end

    return tdst
end

function Base.convert(
    ::Type{TT}, t::AbstractBlockTensorMap
) where {TT<:AbstractBlockTensorMap}
    tdst = similar(TT, space(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = v
    end
    return tdst
end

TensorKit.TensorMap(t::AbstractBlockTensorMap) = convert(TensorMap, t)

# Sparsity
# --------
nonzero_pairs(t::AbstractBlockTensorMap) = nonzero_pairs(parent(t))
nonzero_keys(t::AbstractBlockTensorMap) = nonzero_keys(parent(t))
nonzero_values(t::AbstractBlockTensorMap) = nonzero_values(parent(t))
nonzero_length(t::AbstractBlockTensorMap) = nonzero_length(parent(t))

nonzero_values(A::AbstractArray) = values(A)
nonzero_keys(A::AbstractArray) = keys(A)
nonzero_pairs(A::AbstractArray) = pairs(A)
nonzero_length(A::AbstractArray) = length(A)

issparse(t::AbstractTensorMap) = false
issparse(t::TensorKit.AdjointTensorMap) = issparse(parent(t))

# Show
# ----
function Base.show(io::IO, t::AbstractBlockTensorMap)
    summary(io, t)
    get(io, :compact, false) && return nothing
    println(io, ":")
    for (c, b) in TensorKit.blocks(t)
        println(io, "* Block for sector $c:")
        show(io, b)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", t::AbstractBlockTensorMap)
    # header:
    summary(io, t)
    nnz = nonzero_length(t)
    println(
        io, " with ", nnz, " stored entr", isone(nnz) ? "y" : "ies", iszero(nnz) ? "" : ":"
    )

    # body:
    compact = get(io, :compact, false)::Bool
    (iszero(nnz) || compact) && return nothing
    if issparse(t)
        show_braille(io, t)
    else
        show_elements(io, t)
    end

    return nothing
end

function show_elements(io::IO, x::AbstractBlockTensorMap)
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
function show_braille(io::IO, x::AbstractBlockTensorMap)
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
