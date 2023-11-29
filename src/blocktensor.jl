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
    return BlockTensorMap{S,N₁,N₂,T,N₁+N₂}(args...)
end
function BlockTensorMap{S,N₁,N₂,T,N}(::UndefInitializer, V::TensorMapSumSpace{S,N₁,N₂}) where {S,N₁,N₂,T,N}
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
@propagate_inbounds function Base.setindex!(t::BlockTensorArray{T₁,N}, v::BlockTensorArray{T₂,N}, I::CartesianIndex{N}) where {T₁,T₂,N}
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
        println(io,  " with ", nnz, " stored entr", nnz == 1 ? "y" : "ies", ":")
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

function Base.convert(::Type{BlockTensorMap{S,N₁,N₂,T₁,N}}, t::BlockTensorMap{S,N₁,N₂,T₂,N}) where {S,N₁,N₂,T₁,T₂,N}
    T₁ === T₂ && return t
    tdst = BlockTensorMap{S,N₁,N₂,T₁,N}(undef, codomain(t), domain(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = convert(T₁, v)
    end
    return tdst
end

function Base.convert(::Type{BlockTensorMap}, t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    tdst = BlockTensorMap{S,N₁,N₂,typeof(t)}(undef, convert(ProductSumSpace{S,N₁}, codomain(t)), convert(ProductSumSpace{S,N₂}, domain(t)))
    tdst[1] = t
    return tdst
end

# TensorKit Interface
# -------------------

TK.spacetype(::Union{T,Type{T}}) where {S,T<:BlockTensorMap{S}} = S
function TK.sectortype(::Union{T,Type{T}}) where {S,T<:BlockTensorMap{S}}
    return sectortype(S)
end
TK.storagetype(::Union{B,Type{B}}) where {T,B<:BlockTensorArray{T}} = storagetype(T)
TK.storagetype(::Type{Union{A,B}}) where {A,B} = Union{storagetype(A),storagetype(B)}
function TK.similarstoragetype(TT::Type{<:BlockTensorMap}, ::Type{T}) where {T}
    return Core.Compiler.return_type(similar, Tuple{storagetype(TT),Type{T}})
end
TK.similarstoragetype(t::BlockTensorMap, T) = TK.similarstoragetype(typeof(t), T)

TK.numout(::Union{T,Type{T}}) where {S,N₁,T<:BlockTensorMap{S,N₁}} = N₁
TK.numin(::Union{T,Type{T}}) where {S,N₁,N₂,T<:BlockTensorMap{S,N₁,N₂}} = N₂
function TK.numind(::Union{T,Type{T}}) where {S,N₁,N₂,T<:BlockTensorMap{S,N₁,N₂}}
    return N₁ + N₂
end

TK.codomain(t::BlockTensorMap) = t.codom
TK.domain(t::BlockTensorMap) = t.dom
TK.space(t::BlockTensorMap) = codomain(t) ← domain(t)
TK.space(t::BlockTensorMap, i) = space(t)[i]
TK.dim(t::BlockTensorMap) = dim(space(t))

function TK.codomainind(::Union{T,Type{T}}) where {S,N₁,T<:BlockTensorMap{S,N₁}}
    return ntuple(n -> n, N₁)
end
function TK.domainind(::Union{T,Type{T}}) where {S,N₁,N₂,T<:BlockTensorMap{S,N₁,N₂}}
    return ntuple(n -> N₁ + n, N₂)
end
function TK.allind(::Union{T,Type{T}}) where {S,N₁,N₂,T<:BlockTensorMap{S,N₁,N₂}}
    return ntuple(n -> n, N₁ + N₂)
end

function TK.adjointtensorindex(::BlockTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂}
    return ifelse(i <= N₁, N₂ + i, i - N₁)
end

function TK.adjointtensorindices(t::BlockTensorMap, indices::IndexTuple)
    return map(i -> TK.adjointtensorindex(t, i), indices)
end

function TK.adjointtensorindices(t::BlockTensorMap, p::Index2Tuple)
    return TK.adjointtensorindices(t, p[1]), TK.adjointtensorindices(t, p[2])
end



# Linear Algebra
# --------------
Base.:(+)(t::BlockTensorMap, t2::BlockTensorMap) = add(t, t2)
Base.:(-)(t::BlockTensorMap, t2::BlockTensorMap) = add(t, t2, -one(scalartype(t)))
Base.:(*)(t::BlockTensorMap, α::Number) = scale(t, α)
Base.:(*)(α::Number, t::BlockTensorMap) = scale(t, α)
Base.:(/)(t::BlockTensorMap, α::Number) = scale(t, inv(α))
Base.:(\)(α::Number, t::BlockTensorMap) = scale(t, inv(α))

# TODO: make this lazy?
function Base.adjoint(t::BlockTensorMap)
    tdst = similar(t, domain(t) ← codomain(t))
    adjoint_inds = TO.linearize((domainind(t), codomainind(t)))
    for (I, v) in nonzero_pairs(t)
        I′ = CartesianIndex(getindices(I.I, adjoint_inds)...)
        tdst[I′] = adjoint(v)
    end
    return tdst
end

function LinearAlgebra.axpy!(α::Number, t1::BlockTensorMap, t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    for (i, v) in nonzero_pairs(t1)
        t2[i] = axpy!(α, v, t2[i])
    end
    return t2
end

function LinearAlgebra.axpby!(α::Number, t1::BlockTensorMap, β::Number,
                              t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    rmul!(t2, β)
    for (i, v) in nonzero_pairs(t1)
        t2[i] = axpy!(α, v, t2[i])
    end
    return t2
end

function LinearAlgebra.dot(t1::BlockTensorMap, t2::BlockTensorMap)
    size(t1) == size(t2) || throw(DimensionMismatch("dot arguments have different size"))

    s = zero(promote_type(scalartype(t1), scalartype(t2)))
    if nonzero_length(t1) >= nonzero_length(t2)
        @inbounds for (I, v) in nonzero_pairs(t1)
            s += dot(v, t2[I])
        end
    else
        @inbounds for (I, v) in nonzero_pairs(t2)
            s += dot(t1[I], v)
        end
    end
    return s
end

function LinearAlgebra.mul!(C::BlockTensorMap, α::Number, A::BlockTensorMap)
    space(C) == space(A) || throw(SpaceMismatch())
    SparseArrayKit._zero!(parent(C))
    for (i, v) in nonzero_pairs(A)
        C[i] = mul!(C[i], α, v)
    end
    return C
end

function LinearAlgebra.lmul!(α::Number, t::BlockTensorMap{S}) where {S<:IndexSpace}
    for v in nonzero_values(t)
        lmul!(α, v)
    end
    return t
end

function LinearAlgebra.rmul!(t::BlockTensorMap{S}, α::Number) where {S<:IndexSpace}
    for v in nonzero_values(t)
        rmul!(v, α)
    end
    return t
end

function LinearAlgebra.norm(tA::BlockTensorMap{S,N₁,N₂,A},
                            p::Real=2) where {S,N₁,N₂,A}
    vals = nonzero_values(tA)
    isempty(vals) && return norm(zero(scalartype(tA)), p)
    return LinearAlgebra.norm(norm.(vals), p)
end

function Base.real(t::BlockTensorMap)
    if isreal(sectortype(spacetype(t)))
        t′ = TensorMap(undef, real(scalartype(t)), codomain(t), domain(t))
        for (k, v) in nonzero_pairs(t)
            t′[k] = real(v)
        end

        return t′
    else
        msg = "`real` has not been implemented for `BlockTensorMap{$(S)}`."
        throw(ArgumentError(msg))
    end
end

function Base.imag(t::BlockTensorMap)
    if isreal(sectortype(spacetype(t)))
        t′ = TensorMap(undef, real(scalartype(t)), codomain(t), domain(t))
        for (k, v) in nonzero_pairs(t)
            t′[k] = imag(v)
        end

        return t′
    else
        msg = "`imag` has not been implemented for `BlockTensorMap{$(S)}`."
        throw(ArgumentError(msg))
    end
end

# VectorInterface
# ---------------

function VI.zerovector(t::BlockTensorMap, ::Type{S}) where {S<:Number}
    return similar(t, S, space(t))
end
VI.zerovector!(t::BlockTensorMap) = (empty!(t.data); t)
VI.zerovector!!(t::BlockTensorMap) = zerovector!(t)

function VI.scale(t::BlockTensorMap, α::Number)
    t′ = zerovector(t, VI.promote_scale(t, α))
    scale!(t′, t, α)
    return t′
end
function VI.scale!(t::BlockTensorMap, α::Number)
    for v in nonzero_values(parent(t))
        scale!(v, α)
    end
    return t
end
function VI.scale!(ty::BlockTensorMap, tx::BlockTensorMap,
                                α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch())
    # delete all entries in ty that are not in tx
    for I in setdiff(nonzero_keys(ty), nonzero_keys(tx))
        delete!(ty.data, I)
    end
    # in-place scale elements from tx (getindex might allocate!)
    for (I, v) in nonzero_pairs(tx)
        ty[I] = scale!(ty[I], v, α)
    end
    return ty
end
function VI.scale!!(x::BlockTensorMap, α::Number)
    α === One() && return x
    return VI.promote_scale(x, α) <: scalartype(x) ? scale!(x, α) : scale(x, α)
end
function VI.scale!!(y::BlockTensorMap, x::BlockTensorMap,
                                 α::Number)
    return VI.promote_scale(x, α) <: scalartype(y) ? scale!(y, x, α) : scale(x, α)
end

function VI.add(y::BlockTensorMap, x::BlockTensorMap, α::Number,
                             β::Number)
    space(y) == space(x) || throw(SpaceMisMatch())
    T = VI.promote_add(y, x, α, β)
    z = zerovector(y, T)
    # TODO: combine these operations where possible
    scale!(z, y, β)
    return add!(z, x, α)
end
function VI.add!(y::BlockTensorMap, x::BlockTensorMap, α::Number,
                              β::Number)
    space(y) == space(x) || throw(SpaceMisMatch())
    # TODO: combine these operations where possible
    scale!(y, β)
    for (k, v) in nonzero_pairs(x)
        y[k] = add!(y[k], v, α)
    end
    return y
end
function VI.add!!(y::BlockTensorMap, x::BlockTensorMap, α::Number,
                               β::Number)
    return promote_add(y, x, α, β) <: scalartype(y) ? add!(y, x, α, β) : add(y, x, α, β)
end

function VI.inner(x::BlockTensorMap, y::BlockTensorMap)
    space(y) == space(x) || throw(SpaceMismatch())
    s = zero(VI.promote_inner(x, y))
    for I in intersect(nonzero_keys(x), nonzero_keys(y))
        s += inner(x[I], y[I])
    end
    return s
end

# TODO: this is type-piracy!
# VI.scalartype(::Type{Union{A,B}}) where {A,B} = Union{scalartype(A), scalartype(B)}

# TensorOperations
# ----------------

function TO.tensoradd_type(TC, ::Index2Tuple{N₁,N₂}, ::BlockTensorMap{S},
                           conjA::Symbol) where {S,N₁,N₂}
    T = tensormaptype(S, N₁, N₂, TC)
    return BlockTensorMap{S,N₁,N₂,T,N₁ + N₂}
end

function TO.tensoradd_structure(pC::Index2Tuple{N₁,N₂}, A::BlockTensorMap{S},
                                conjA::Symbol) where {S,N₁,N₂}
    if conjA == :N
        pC′ = pC
        V = space(A)
    else
        pC′ = TK.adjointtensorindices(A, pC)
        V = space(A)'
    end
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(V), pC′[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(V), pC′[2])))
    return dom → cod
end

function TO.tensorcontract_type(TC::Type{<:Number}, pC::Index2Tuple{N₁,N₂},
                                A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                                istemp=false, backend::Backend...) where {S,N₁,N₂}
    M = TK.similarstoragetype(A, TC)
    M == TK.similarstoragetype(B, TC) ||
        throw(ArgumentError("incompatible storage types"))
    T = tensormaptype(S, N₁, N₂, M)
    return BlockTensorMap{S,N₁,N₂,T,N₁ + N₂}
end

function TO.tensorcontract_structure(pC::Index2Tuple{N₁,N₂}, A::BlockTensorMap{S},
                                     pA::Index2Tuple, conjA::Symbol,
                                     B::BlockTensorMap{S},
                                     pB::Index2Tuple, conjB::Symbol) where {S,N₁,N₂}
    spaces1 = TO.flag2op(conjA).(space.(Ref(A), pA[1]))
    spaces2 = TO.flag2op(conjB).(space.(Ref(B), pB[2]))
    spaces = (spaces1..., spaces2...)
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(spaces), pC[2])))
    return dom → cod
end

function TO.tensoradd!(C::BlockTensorMap{S}, pC::Index2Tuple,
                       A::BlockTensorMap{S}, conjA::Symbol,
                       α::Number, β::Number) where {S}
    argcheck_tensoradd(C, pC, A)
    dimcheck_tensoradd(C, pC, A)

    scale!(C, β)
    indCinA = linearize(pC)
    for (IA, v) in nonzero_pairs(A)
        IC = CartesianIndex(TupleTools.getindices(IA.I, indCinA))
        C[IC] = tensoradd!(C[IC], pC, v, conjA, α, One())
    end
    return C
end

function TO.tensorcontract!(C::BlockTensorMap{S}, pC::Index2Tuple,
                            A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number) where {S}
    argcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)
    dimcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)

    scale!(C, β)

    keysA = sort!(collect(nonzero_keys(A));
                  by=IA -> CartesianIndex(getindices(IA.I, pA[2])))
    keysB = sort!(collect(nonzero_keys(B));
                  by=IB -> CartesianIndex(getindices(IB.I, pB[1])))

    iA = iB = 1
    @inbounds while iA <= length(keysA) && iB <= length(keysB)
        IA = keysA[iA]
        IB = keysB[iB]
        IAc = CartesianIndex(getindices(IA.I, pA[2]))
        IBc = CartesianIndex(getindices(IB.I, pB[1]))
        if IAc == IBc
            Ic = IAc
            jA = iA
            while jA < length(keysA)
                if CartesianIndex(getindices(keysA[jA + 1].I, pA[2])) == Ic
                    jA += 1
                else
                    break
                end
            end
            jB = iB
            while jB < length(keysB)
                if CartesianIndex(getindices(keysB[jB + 1].I, pB[1])) == Ic
                    jB += 1
                else
                    break
                end
            end
            rA = iA:jA
            rB = iB:jB
            if length(rA) < length(rB)
                for kB in rB
                    IB = keysB[kB]
                    IBo = CartesianIndex(getindices(IB.I, pB[2]))
                    vB = B[IB]
                    for kA in rA
                        IA = keysA[kA]
                        IAo = CartesianIndex(getindices(IA.I, pA[1]))
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(getindices(IABo.I, linearize(pC)))
                        vA = A[IA]
                        C[IC] = tensorcontract!(C[IC], pC, vA, pA, conjA, vB, pB, conjB, α,
                                                One())
                    end
                end
            else
                for kA in rA
                    IA = keysA[kA]
                    IAo = CartesianIndex(getindices(IA.I, pA[1]))
                    vA = A[IA]
                    for kB in rB
                        IB = keysB[kB]
                        IBo = CartesianIndex(getindices(IB.I, pB[2]))
                        vB = parent(B).data[IB]
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(getindices(IABo.I, linearize(pC)))
                        C[IC] = tensorcontract!(C[IC], pC, vA, pA, conjA, vB, pB, conjB, α,
                                                One())
                    end
                end
            end
            iA = jA + 1
            iB = jB + 1
        elseif IAc < IBc
            iA += 1
        else
            iB += 1
        end
    end

    return C
end

function TO.tensortrace!(C::BlockTensorMap{S}, pC::Index2Tuple,
                         A::BlockTensorMap{S},
                         pA::Index2Tuple,
                         conjA::Symbol, α::Number, β::Number) where {S}
    argcheck_tensortrace(C, pC, A, pA)
    dimcheck_tensortrace(C, pC, A, pA)

    scale!(C, β)

    for (IA, v) in nonzero_pairs(A)
        IAc1 = CartesianIndex(getindices(IA.I, pA[1]))
        IAc2 = CartesianIndex(getindices(IA.I, pA[2]))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(getindices(IA.I, linearize(pC)))
        C[IC] = tensortrace!(C[IC], pC, v, pA, conjA, α, one(β))
    end
    return C
end

function TO.tensorscalar(C::BlockTensorArray{T,0}) where {T}
    return isempty(C.data) ? zero(scalartype(C)) : tensorscalar(C[])
end

TO.tensorstructure(t::BlockTensorMap) = space(t)
function TO.tensorstructure(t::BlockTensorMap, iA::Int, conjA::Symbol)
    return conjA == :N ? space(t, iA) : conj(space(t, iA))
end

function TO.checkcontractible(tA::BlockTensorMap{S}, iA::Int, conjA::Symbol,
                              tB::BlockTensorMap{S}, iB::Int, conjB::Symbol,
                              label) where {S}
    sA = TO.tensorstructure(tA, iA, conjA)'
    sB = TO.tensorstructure(tB, iB, conjB)
    sA == sB ||
        throw(SpaceMismatch("incompatible spaces for $label: $sA ≠ $sB"))
    return nothing
end

# methods for automatically working with TensorMap - BlockTensorMaps
# ------------------------------------------------------------------------

for (T1, T2) in
    ((:AbstractTensorMap, :BlockTensorMap), (:BlockTensorMap, :AbstractTensorMap),
     (:BlockTensorMap, :BlockTensorMap), (:AbstractTensorMap, :AbstractTensorMap))
    if T1 !== :AbstractTensorMap && T2 !== :AbstractTensorMap
        @eval function TO.tensorcontract!(C::AbstractTensorMap, pC::Index2Tuple, A::$T1,
                                          pA::Index2Tuple, conjA::Symbol, B::$T2,
                                          pB::Index2Tuple, conjB::Symbol, α, β::Number,
                                          backend::TO.Backend...)
            C′ = convert(BlockTensorMap, C)
            tensorcontract!(C′, pC, A, pA, conjA, B, pB, conjB, α, β, backend...)
            return C
        end
        
        @eval function TO.checkcontractible(tA::$T1, iA::Int, conjA::Symbol,
                                            tB::$T2, iB::Int, conjB::Symbol,
                                            label)
            sA = TO.tensorstructure(tA, iA, conjA)'
            sB = TO.tensorstructure(tB, iB, conjB)
            sA == sB ||
                throw(SpaceMismatch("incompatible spaces for $label: $sA ≠ $sB"))
            return nothing
        end
    end

    if T1 !== T2
        @eval function TO.tensorcontract_type(TC, pC, A::$T1, pA, conjA, B::$T2, pB, conjB)
            return TO.tensorcontract_type(TC, pC, convert(BlockTensorMap, A), pA, conjA,
                                          convert(BlockTensorMap, B), pB, conjB)
        end
        
        @eval function TO.tensorcontract_structure(pC::Index2Tuple{N₁,N₂}, A::$T1{S},
                                             pA::Index2Tuple, conjA::Symbol,
                                             B::$T2{S},
                                             pB::Index2Tuple, conjB::Symbol) where {S,N₁,N₂}
            spaces1 = TO.flag2op(conjA).(space.(Ref(A), pA[1]))
            spaces2 = TO.flag2op(conjB).(space.(Ref(B), pB[2]))
            spaces = (spaces1..., spaces2...)
            cod = ProductSumSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
            dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(spaces), pC[2])))
            return dom → cod
        end
    end

    if !(T1 === :BlockTensorMap && T2 === :BlockTensorMap)
        @eval function TO.tensorcontract!(C::BlockTensorMap, pC::Index2Tuple, A::$T1,
                                          pA::Index2Tuple, conjA::Symbol, B::$T2,
                                          pB::Index2Tuple, conjB::Symbol, α::Number,
                                          β::Number, backend::TO.Backend...)
            return TO.tensorcontract!(C, pC, convert(BlockTensorMap, A), pA, conjA,
                                      convert(BlockTensorMap, B), pB, conjB, α, β,
                                      backend...)
        end
    end
end

# TODO: similar for tensoradd!, tensortrace!

Base.haskey(t::BlockTensorMap, I::CartesianIndex) = haskey(t.data, I)
function Base.haskey(t::BlockTensorMap, i::Int)
    return haskey(t.data, CartesianIndices(t)[i])
end
