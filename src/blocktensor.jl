struct BlockTensorMap{S<:IndexSpace,N₁,N₂,A<:AbstractArray{<:AbstractTensorMap{S,N₁,N₂}}}
    data::A
    cod::ProductSumSpace{S,N₁}
    dom::ProductSumSpace{S,N₂}
end

function BlockTensorMap(blocks::AbstractArray{<:AbstractTensorMap{S}},
                        cod::TensorSumSpace{S}, dom::TensorSumSpace{S}) where {S}
    return BlockTensorMap(blocks, convert(ProductSpace, cod), convert(ProductSpace, dom))
end
function BlockTensorMap(blocks::AbstractArray{<:AbstractTensorMap{S}},
                        V::TensorKit.HomSpace{SumSpace{S}}) where {S}
    return BlockTensorMap(blocks, codomain(V), domain(V))
end

# undef_blocks initializer: uninitialized blocks
#-----------------------------------------------
function BlockTensorMap(::BlockArrays.UndefBlocksInitializer,
                        blocktype::Type{<:AbstractArray{T}}, cod::ProductSumSpace{S,N₁},
                        dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂,
                                                           T<:AbstractTensorMap{S,N₁,N₂}}
    sz = ntuple(i -> i > N₁ ? length(dom[i - N₁]) : length(cod[i]), N₁ + N₂)
    data = blocktype(undef, sz)
    return BlockTensorMap(data, cod, dom)
end
function BlockTensorMap(::BlockArrays.UndefBlocksInitializer,
                        tensortype::Type{<:AbstractTensorMap{S,N₁,N₂}},
                        cod::ProductSumSpace{S,N₁},
                        dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
    return BlockTensorMap(undef_blocks, Array{tensortype,N₁ + N₂}, cod, dom)
end
function BlockTensorMap(::BlockArrays.UndefBlocksInitializer, scalartype::Type{<:Number},
                        cod::ProductSumSpace{S,N₁},
                        dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
    return BlockTensorMap(undef_blocks, tensormaptype(S, N₁, N₂, scalartype), cod, dom)
end
function BlockTensorMap(::BlockArrays.UndefBlocksInitializer,
                        cod::ProductSumSpace{S,N₁},
                        dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
    return BlockTensorMap(undef_blocks, tensormaptype(S, N₁, N₂), cod, dom)
end
function BlockTensorMap(::BlockArrays.UndefBlocksInitializer,
                        cod::TensorSumSpace{S},
                        dom::TensorSumSpace{S}) where {S}
    return BlockTensorMap(undef_blocks, convert(ProductSpace, cod),
                          convert(ProductSpace, dom))
end
function BlockTensorMap(::BlockArrays.UndefBlocksInitializer,
                        ::Type{T}, cod::TensorSumSpace{S},
                        dom::TensorSumSpace{S}) where {S,T}
    return BlockTensorMap(undef_blocks, T, convert(ProductSpace, cod),
                          convert(ProductSpace, dom))
end

# undef initializer: initialized blocks, uninitialized tensors
# ------------------------------------------------------------
function BlockTensorMap(::UndefInitializer,
                        blocktype::Type{<:AbstractArray{<:AbstractTensorMap{S,N₁,N₂}}},
                        cod::ProductSumSpace{S,N₁},
                        dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
    t = BlockTensorMap(undef_blocks, blocktype, cod, dom)
    for I in CartesianIndices(t.data)
        t.data[I] = TensorMap(undef, scalartype(t), getsubspace(space(t), I))
    end
    return t
end
function BlockTensorMap(::UndefInitializer, scalartype::Type{<:Number},
                        cod::ProductSpace{S,N₁},
                        dom::ProductSpace{S,N₂}) where {S<:SumSpace,N₁,N₂}
    t = BlockTensorMap(undef_blocks, scalartype, cod, dom)
    for I in CartesianIndices(t.data)
        t.data[I] = TensorMap(undef, scalartype, getsubspace(space(t), I))
    end
    return t
end
function BlockTensorMap(::UndefInitializer,
                        cod::ProductSpace{S,N₁},
                        dom::ProductSpace{S,N₂}) where {S<:SumSpace,N₁,N₂}
    t = BlockTensorMap(undef_blocks, cod, dom)
    for I in CartesianIndices(t.data)
        t.data[I] = TensorMap(undef, scalartype(t), getsubspace(space(t), I))
    end
    return t
end

# HomSpace constructors
# ---------------------
function BlockTensorMap(f, V::TensorKit.HomSpace{SumSpace{S}}) where {S}
    return BlockTensorMap(f, codomain(V), domain(V))
end
function BlockTensorMap(f, ::Type{T}, V::TensorKit.HomSpace{SumSpace{S}}) where {S,T}
    return BlockTensorMap(f, T, codomain(V), domain(V))
end

#==============
TODO: the following converting mechanism is a bit dubious. There is a choice to be made,
if we want to concatenate tensors using their matrix representations, or if we want to
concatenate them using their tensor representations. The former is more efficient, but the
latter is more intuitive. The current implementation uses the matrix representation, as this
is useful in testing scenarios, but it is not clear if this is the best choice.
==============#
function BlockTensorMap(tsrc::TensorMap{S,N₁,N₂}, cod::ProductSumSpace{S,N₁},
                        dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
    tdst = BlockTensorMap(undef, typeof(tsrc), cod, dom)

    for (c, b) in tdst
        block_szs = block_sizes(b)
        pseudoblock = PseudoBlockArray(block(tsrc, c), block_szs...)
        for I in CartesianIndices(size(tdst))
            b[I] .= pseudoblock[Block(I.I...)]
        end
    end

    return tdst
end
function BlockTensorMap(tsrc::TensorMap, spaces::TensorKit.HomSpace)
    return BlockTensorMap(tsrc, codomain(spaces), domain(spaces))
end

# from data: extract the spaces
# -----------------------------
# This is technically type-piracy?
function BlockArrays.mortar(blocks::AbstractArray{<:AbstractTensorMap})
    return mortar(blocks, spaces_from_blocks(blocks)...)
end

function spaces_from_blocks(blocks::AbstractArray{T}) where {T<:AbstractTensorMap}
    S = spacetype(T)
    N₁ = numout(T)
    N₂ = numin(T)
    
    length(blocks) == 0 &&
        return ProductSumSpace{S,N₁}(ntuple(i -> SumSpace{S}(), N₁)),
               ProductSumSpace{S,N₂}(ntuple(i -> SumSpace{S}(), N₂))

    fullspaces = map!(space, Array{S,ndims(blocks)}(undef, size(blocks)), blocks)
    ranges = ntuple(i => 1:stride(fullspaces, i):size(fullspaces, i), N₁ + N₂)
    cod = ProductSumSpace{S,N₁}(ntuple(i -> SumSpace(fullspaces[ranges[i]]), N₁))
    dom = ProductSumSpace{S,N₂}(ntuple(i -> SumSpace(fullspaces[ranges[N₁ + i]])', N₂))
    _checkspaces(fullspaces, cod, dom)
    return cod, dom
end

function _checkspaces(fullspaces, cod, dom)
    homspace = cod ← dom
    for I in CartesianIndices(size(fullspaces))
        fullspaces[I] == getsubspace(homspace, I) ||
            throw(DomainError("The spaces of the blocks are not subspaces of the codomain and domain"))
    end
end

# similar
# -------
function Base.similar(::BlockTensorMap{S}, ::Type{T},
                      P::TensorKit.TensorMapSpace{SumSpace{S}}) where {T<:Number,S}
    return BlockTensorMap(undef_blocks, T, codomain(P), domain(P))
end

# function TensorKit.TensorMap(data::AbstractArray{<:Number}, cod::ProductSumSpace{S,N₁},
#                              dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
#     tmp = TensorMap(data, mapreduce(join, ⊗, cod.spaces; init=ProductSpace{S′,0}()),
#                     mapreduce(join, ⊗, dom.spaces; init=ProductSpace{S′,0}()))
#     return TensorMap(tmp, cod ← dom)
# end

# function TensorKit.TensorMap(::UndefInitializer, T::Type{<:Number},
#                              cod::ProductSumSpace{S,N₁},
#                              dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
#     return BlockTensorMap(undef, T, cod, dom)
# end

# function TensorKit.TensorMap(f, cod::ProductSumSpace{S,N₁},
#                              dom::ProductSumSpace{S,N₂}) where {S,N₁,N₂}
#     T = eltype(f((1, 1)))
#     t = TensorMap(undef, T, cod, dom)
#     for I in CartesianIndices(t.data)
#         t[I] = TensorMap(f, getsubspace(space(t), I))
#     end
#     return t
# end

# Special-purpose constructors
# ----------------------------
function TensorKit.one!(t::BlockTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("no identity if domain and codomain are different"))
    cod_inds = codomainind(t)
    dom_inds = domainind(t)
    for I in eachindex(t)
        if getindices(I.I, cod_inds) == getindices(I.I, dom_inds)
            one!(t[I])
        else
            zerovector!(t[I])
        end
    end
    return t
end

function TensorKit.isomorphism(::Type{A}, cod::ProductSpace{S},
                               dom::ProductSpace{S}) where {A<:DenseMatrix,S<:SumSpace}
    cod ≅ dom || throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic"))
    t = TensorMap(undef, scalartype(A), cod, dom)

    cartesian_rowinds = CartesianIndices(length.(codomain(t).spaces))
    cartesian_colinds = CartesianIndices(length.(domain(t).spaces))

    sz = size(t)
    rows = prod(getindices(sz, codomainind(t)))
    cols = prod(getindices(sz, domainind(t)))

    for c in blocksectors(t)
        rowdims = cumsum(ntuple(rows) do i
                             return prod(blockdim.(getindex.(codomain(t).spaces,
                                                             cartesian_rowinds[i].I),
                                                   Ref(c)))
                         end)
        rowranges = UnitRange.((0, Base.front(rowdims)...) .+ 1, rowdims)
        coldims = cumsum(ntuple(cols) do i
                             return prod(blockdim.(getindex.(domain(t).spaces,
                                                             cartesian_colinds[i].I),
                                                   Ref(c)))
                         end)
        colranges = UnitRange.((0, Base.front(coldims)...) .+ 1, coldims)
        for i in 1:rows, j in 1:cols
            if first(colranges[j]) <= last(rowranges[i]) <= last(colranges[j]) ||
               first(rowranges[i]) <= last(colranges[j]) <= last(rowranges[i])
                t′ = t[i + (j - 1) * cols]
                copyto!(block(t′, c),
                        reshape(rowranges[i], :, 1) .== reshape(colranges[j]', 1, :))
                t[i + (j - 1) * cols] = t′
            end
        end
    end
    return t
end

function TensorKit.isometry(::Type{A}, cod::ProductSpace{S},
                            dom::ProductSpace{S}) where {A<:DenseMatrix,S<:SumSpace}
    InnerProductStyle(S) === EuclideanProduct() ||
        throw(ArgumentError("isometries require Euclidean inner product"))
    dom ≾ cod ||
        throw(SpaceMismatch("codomain $cod and domain $dom do not allow for an isometric mapping"))
    t = TensorMap(undef, scalartype(A), cod, dom)
    cartesian_rowinds = CartesianIndices(length.(codomain(t).spaces))
    cartesian_colinds = CartesianIndices(length.(domain(t).spaces))

    sz = size(t)
    rows = prod(getindices(sz, codomainind(t)))
    cols = prod(getindices(sz, domainind(t)))

    for c in blocksectors(t)
        rowdims = cumsum(ntuple(rows) do i
                             return prod(blockdim.(getindex.(codomain(t).spaces,
                                                             cartesian_rowinds[i].I),
                                                   Ref(c)))
                         end)
        rowranges = UnitRange.((0, Base.front(rowdims)...) .+ 1, rowdims)
        coldims = cumsum(ntuple(cols) do i
                             return prod(blockdim.(getindex.(domain(t).spaces,
                                                             cartesian_colinds[i].I),
                                                   Ref(c)))
                         end)
        colranges = UnitRange.((0, Base.front(coldims)...) .+ 1, coldims)
        for i in 1:rows, j in 1:cols
            if first(colranges[j]) <= last(rowranges[i]) <= last(colranges[j]) ||
               first(rowranges[i]) <= last(colranges[j]) <= last(rowranges[i])
                t′ = t[i + (j - 1) * cols]
                copyto!(block(t′, c),
                        reshape(rowranges[i], :, 1) .== reshape(colranges[j]', 1, :))
                t[i + (j - 1) * cols] = t′
            end
        end
    end
    return t
end

function Base.adjoint(t::BlockTensorMap{S,N₁,N₂,A}) where {S,N₁,N₂,A}
    T = TensorKit.adjointtensormaptype(eltype(spacetype(t)), numin(t), numout(t),
                                       storagetype(t))
    tdst = BlockTensorMap(undef_blocks, T, domain(t), codomain(t))
    adjoint_inds = [domainind(t)..., codomainind(t)...]
    for I in eachindex(t)
        I′ = CartesianIndex(I.I[adjoint_inds])
        tdst[I′] = adjoint(t[I])
    end
    return tdst
end

# function TensorKit.permute(t::BlockTensorMap{S,<:Any,<:Any,A}, p1::IndexTuple{N₁}, p2::IndexTuple{N₂}; copy::Bool=false) where {S,N₁,N₂,A}
#     cod = ProductSpace{S,N₁}(map(n -> space(t, n), p1))
#     dom = ProductSpace{S,N₂}(map(n -> dual(space(t, n)), p2))

#     if !copy && p1 === codomainind(t) && p2 === domainind(t)
#         return t
#     end

#     tdst = TensorMap(undef_blocks, scalartype(t), cod, dom)

#     for I in eachindex(t)
#         tdst[I.I[p1..., p2...]] = permute(t[I], p1, p2; copy=copy)
#     end
#     return tdst
# end

# function TensorKit._add!(α, tsrc::BlockTensorMap{S}, β, tdst::BlockTensorMap{S,N₁,N₂},
#                p1::IndexTuple{N₁}, p2::IndexTuple{N₂}, fusiontreemap) where {S,N₁,N₂}
#     @boundscheck begin
#         all(i -> space(tsrc, p1[i]) == space(tdst, i), 1:N₁) ||
#             throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)),
#             tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
#         all(i -> space(tsrc, p2[i]) == space(tdst, N₁ + i), 1:N₂) ||
#             throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)),
#             tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
#     end

#     isone(β) || rmul!(tdst, β)

#     p = (p1..., p2...)
#     for (key, val) in nonzero_pairs(tsrc)
#         tdst[CartesianIndex(getindices(key.I, p))] = _add!(α, val, true,
#                                                            tdst[CartesianIndex(getindices(key.I,
#                                                                                           p))],
#                                                            p1, p2, fusiontreemap)
#     end
#     return tdst
# end

#===========================================================================================
    AbstractArray
===========================================================================================#

Base.parent(A::BlockTensorMap) = A.data
Base.eltype(::Type{<:BlockTensorMap{<:Any,<:Any,<:Any,A}}) where {A} = eltype(A)
Base.size(A::BlockTensorMap) = size(parent(A))
Base.length(A::BlockTensorMap) = length(parent(A))
Base.ndims(A::BlockTensorMap) = ndims(parent(A))
Base.axes(A::BlockTensorMap) = axes(parent(A))
Base.axes(A::BlockTensorMap, i::Int) = axes(parent(A), i)

@inline Base.checkbounds(a::BlockTensorMap, I...) = checkbounds(parent(a), I...)

# scalar indexing yields a TensorMap
function Base.getindex(A::BlockTensorMap, I::CartesianIndex)
    if A isa SparseBlockTensorMap
        @boundscheck checkbounds(A, I)
        return get(parent(A).data, I) do
            return TensorMap(zeros, scalartype(A), getsubspace(space(A), I))
        end
    else
        return getindex(parent(A), I)
    end
end
Base.getindex(A::BlockTensorMap, I::Vararg{Int}) = getindex(A, CartesianIndex(I))

# non-scalar indexing yields a BlockTensorMap
function Base.getindex(A::BlockTensorMap, I...)
    tensors = reshape(getindex(parent(A), I...), length.(I))
    return BlockTensorMap(tensors, getsubspace(space(A), I...))
end

function Base.setindex!(A::BlockTensorMap, v, I...)
    return setindex!(parent(A), v, I...)
end
function Base.setindex!(A::BlockTensorMap, v::BlockTensorMap, I...)
    return setindex!(parent(A), parent(v), I...)
end
function Base.setindex!(A::BlockTensorMap, v::BlockTensorMap, I::Vararg{Int})
    @assert length(v) == 1
    return setindex!(parent(A), v[1], I...)
end

Base.firstindex(A::BlockTensorMap, args...) = firstindex(parent(A), args...)
Base.lastindex(A::BlockTensorMap, args...) = lastindex(parent(A), args...)

# @inline function Base.getindex(A::BlockTensorMap, I::CartesianIndex{N}) where {N}
#     ndims(A) == N || throw(ArgumentError("invalid index style"))
#     @boundscheck checkbounds(A, I)
#     return get(parent(A).data, I) do
#         inds = Tuple(I)
#         cod = getsubspace(codomain(A), getindices(inds, codomainind(A))...)
#         dom = getsubspace(domain(A), getindices(inds, domainind(A))...)
#         return TensorMap(zeros, scalartype(A), cod, dom)::eltype(parent(A))
#     end
# end
# Base.@propagate_inbounds function Base.getindex(a::BlockTensorMap{<:Any,<:Any,<:Any,A},
#                                                 I::Vararg{Int,N}) where {N,A<:AbstractArray{<:Any,N}}
#     return getindex(a, CartesianIndex(I))
# end
# Base.@propagate_inbounds function Base.getindex(a::BlockTensorMap, i::Int)
#     return getindex(a, CartesianIndices(size(a))[i])
# end

# @inline function Base.setindex!(a::BlockTensorMap{<:Any,<:Any,<:Any,<:AbstractArray{T,N}}, v::T,I::CartesianIndex{N}) where {T,N}
#     @boundscheck checkbounds(a, I)
#     parent(a).data[I] = v
#     return v
# end
# Base.@propagate_inbounds function Base.setindex!(a::BlockTensorMap{<:Any,<:Any,<:Any,<:AbstractArray{T,N}}, v::T, I::Vararg{Int,N}) where {T,N}
#     return setindex!(a, v, CartesianIndex(I))
# end
# Base.@propagate_inbounds function Base.setindex!(a::BlockTensorMap, v::AbstractTensorMap, i::Int)
#     return setindex!(a, v, CartesianIndices(size(a))[i])
# end

Base.eachindex(a::BlockTensorMap) = CartesianIndices(size(a))
Base.keys(a::BlockTensorMap) = keys(parent(a))

for f in (:nonzero_keys, :nonzero_length, :nonzero_pairs, :nonzero_values)
    @eval SparseArrayKit.$f(A::BlockTensorMap) = $f(parent(A))
end

function Base.copy!(tdst::BlockTensorMap, tsrc::BlockTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch())
    copy!(parent(tdst), parent(tsrc))
    return tdst
end

function Base.:(==)(t1::BlockTensorMap, t2::BlockTensorMap)
    codomain(t1) == codomain(t2) && domain(t1) == domain(t2) || return false
    keys = collect(nonzero_keys(t1))
    intersect!(keys, nonzero_keys(t2))
    if !(length(keys) == length(nonzero_keys(t1)) == length(nonzero_keys(t2)))
        return false
    end
    for I in keys
        t1[I] == t2[I] || return false
    end
    return true
end

function Base.hash(t::BlockTensorMap, h::UInt)
    h = hash(codomain(t), h)
    h = hash(domain(t), h)
    return hash(parent(t), h)
end

# Promotion and conversion
# ------------------------
function Base.promote_rule(::Type{<:BlockTensorMap{S,N₁,N₂}},
                           ::Type{<:AbstractTensorMap{S,N₁,N₂}}) where {S,N₁,N₂}
    return BlockTensorMap{S,N₁,N₂}
end

function Base.convert(::Type{<:BlockTensorMap{S,N₁,N₂}},
                      tsrc::TensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    cod = mapreduce(SumSpace, ⊗, codomain(tsrc); init=ProductSumSpace{S,0}())
    dom = mapreduce(SumSpace, ⊗, domain(tsrc); init=ProductSumSpace{S,0}())
    tdst = BlockTensorMap(reshape([tsrc], ones(Int, N₁ + N₂)...), cod, dom)
    return tdst
end
function Base.convert(::Type{BlockTensorMap{S,N₁,N₂}},
                      tsrc::BlockTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    return tsrc
end
function Base.convert(::Type{BlockTensorMap},
                      tsrc::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    return convert(BlockTensorMap{S,N₁,N₂}, tsrc)
end

# show and friends
#-----------------
function Base.show(io::IO, t::BlockTensorMap)
    if get(io, :compact, false)
        print(io, "BlockTensorMap(", space(t), ")")
        return nothing
    end

    println(io, "BlockTensorMap(", space(t), "):")
    show(io, parent(t))
    return nothing
end

# function Base.show(io::IO, ::MIME"text/plain", x::BlockTensorMap)
#     xnnz = nonzero_length(x)
#     print(io, join(size(x), "×"), " ", typeof(x), " with ", xnnz, " stored ",
#           xnnz == 1 ? "entry" : "entries")
#     if xnnz != 0
#         println(io, ":")
#         show(IOContext(io, :typeinfo => eltype(x)), x)
#     end
# end
# Base.show(io::IO, x::BlockTensorMap) = show(convert(IOContext, io), x)
# function Base.show(io::IOContext, x::BlockTensorMap)
#     nzind = nonzero_keys(x)
#     if isempty(nzind)
#         return show(io, MIME("text/plain"), x)
#     end
#     limit = get(io, :limit, false)::Bool
#     half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
#     pads = map(1:ndims(x)) do i
#         return ndigits(maximum(getindex.(nzind, i)))
#     end
#     if !haskey(io, :compact)
#         io = IOContext(io, :compact => true)
#     end
#     for (k, (ind, val)) in enumerate(nonzero_pairs(x))
#         if k < half_screen_rows || k > length(nzind) - half_screen_rows
#             print(io, "  ", '[', join(lpad.(Tuple(ind), pads), ","), "]  =  ", val)
#             k != length(nzind) && println(io)
#         elseif k == half_screen_rows
#             println(io, "   ", join(" " .^ pads, " "), "   \u22ee")
#         end
#     end
# end

#===========================================================================================
    AbstractTensorMap
===========================================================================================#
TensorKit.domain(t::BlockTensorMap) = t.dom
TensorKit.codomain(t::BlockTensorMap) = t.cod
TensorKit.storagetype(T::Type{<:BlockTensorMap}) = storagetype(eltype(T))
function TensorKit.storagetype(::Type{Union{T1,T2}}) where {T1,T2}
    return Union{storagetype(T1),storagetype(T2)}
end

function TensorKit.tensormaptype(::Type{S}, N₁::Int, N₂::Int,
                                 ::Type{T}) where {S<:SumSpace,T}
    elTensorType = tensormaptype(eltype(S), N₁, N₂, T)
    return BlockTensorMap{S,N₁,N₂,Array{elTensorType,N₁ + N₂}}
end
# function TensorKit.adjointtensormaptype(::Type{S}, N₁::Int, N₂::Int,
#                                        ::Type{T}) where {S<:SumSpace,T}
#     elTensorType = adjointtensormaptype(eltype(S), N₁, N₂, T)
#     return BlockTensorMap{S,N₁,N₂,Array{elTensorType,N₁ + N₂}}
# end

# const TrivialBlockTensorMap{S,N₁,N₂,A} where {A<:AbstractArray{<:Union{TensorKit.TrivialTensorMap, TensorKit.AdjointTrivialTensorMap},N}} = BlockTensorMap{S,N₁,N₂,A}
# const BlockTensor{S,N,A<:AbstractArray} = BlockTensorMap{S,N,0,A}

function TensorKit.blocksectors(t::BlockTensorMap)
    if eltype(t) isa TensorKit.TrivialTensorMap
        return TensorKit.TrivialOrEmptyIterator(TensorKit.dim(t) == 0)
    else
        return blocksectors(codomain(t) ← domain(t))
    end
end

function TensorKit.dim(t::BlockTensorMap)
    return mapreduce(+, blocksectors(t); init=0) do c
        return blockdim(codomain(t), c) * blockdim(domain(t), c)
    end
end

TensorKit.hasblock(t::BlockTensorMap, c::Sector) = c ∈ blocksectors(t)

function TensorKit.block(t::BlockTensorMap, c::Sector)
    rows = prod(getindices(size(t), codomainind(t)))
    cols = prod(getindices(size(t), domainind(t)))

    if rows == 0 || cols == 0
        error("to be added")
    end

    rowdims = subblockdims(codomain(t), c)
    coldims = subblockdims(domain(t), c)

    b = BlockArray{scalartype(t),2,SparseArray{storagetype(t),2}}(undef, rowdims, coldims)
    lin_inds = LinearIndices(parent(t))
    new_cart_inds = CartesianIndices((rows, cols))
    for (i, v) in nonzero_pairs(t)
        b[Block(new_cart_inds[lin_inds[i]].I)] = TensorKit.block(v, c)
    end

    return b
end

TensorKit.blocks(t::BlockTensorMap) = (c => TensorKit.block(t, c) for c in blocksectors(t))

function Base.convert(::Type{<:TensorMap{S,N₁,N₂}},
                      tsrc::BlockTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    length(tsrc) == 1 && return tsrc[1] # special case if trivial wrapper

    cod = convert(ProductSpace{S,N₁}, codomain(tsrc))
    dom = convert(ProductSpace{S,N₂}, domain(tsrc))

    tdst = TensorMap(undef, scalartype(tsrc), cod, dom)

    if sectortype(tsrc) == TensorKit.Trivial
        block_sizes = vcat(map(x -> dim.(x), codomain(tsrc)),
                           map(x -> dim.(x), domain(tsrc)))
        data = PseudoBlockArray(tdst[], block_sizes...)
        for I in eachindex(tsrc)
            data[Block(I.I...)] .= tsrc[I][]
        end
    else
        for (f1, f2) in fusiontrees(tdst)
            block_sizes = map(space(tsrc), (f1.uncoupled..., f2.uncoupled...)) do (V, c)
                return dim.(V.spaces, Ref(c))
            end
            data = PseudoBlockArray(tdst[f1, f2], block_sizes...)
            for I in eachindex(tsrc)
                if length(data[Block(I.I...)]) > 0
                    data[Block(I.I...)] .= tsrc[I][f1, f2]
                end
            end
        end
    end
    return tdst
end
