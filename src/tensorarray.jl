struct BlockTensorMap{S<:SumSpace,N₁,N₂,T<:AbstractTensorMap,N} <: AbstractTensorMap{S,N₁,N₂}
    data::SparseArray{T,N}
    cod::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}

    function BlockTensorMap{S,N₁,N₂,T,N}(data::SparseArray{T,N}, cod::ProductSpace{S,N₁},
                                                dom::ProductSpace{S,N₂}) where {S,N₁,N₂,T,N}
        N₁ + N₂ == N || throw(ArgumentError("invalid partition $N ≠ $N₁ + $N₂"))
        N₁ == numout(T) && N₂ == numin(T) ||
            throw(ArgumentError("invalid number of indices ($N₁,$N₂) ≠ ($(numout(T)),$(numin(T)))"))
        sumspacetype(spacetype(T)) === S ||
            throw(ArgumentError("invalid spacetype $S and $(spacetype(T))"))
        return new{S,N₁,N₂,T,N}(data, cod, dom)
    end
end

function BlockTensorMap(data::AbstractArray{T,N}, cod::ProductSpace{S,N₁},
                             dom::ProductSpace{S,N₂}) where {S′,𝕂,S<:SumSpace{𝕂,S′},N₁,N₂,
                                                             T<:AbstractTensorMap{S′,N₁,N₂},
                                                             N}
    return TensorAbstractArray{S,N₁,N₂,T,N}(convert(SparseArray{T,N}, data), cod, dom)
end

function BlockTensorMap(data::AbstractArray{T}) where {T<:AbstractTensorMap}
    spaces = _extract_spaces(data)
    codomain = ProductSpace{S}(getindex.((spaces,), codomainind(T)))
    domain = ProductSpace{S}(dual.(getindex.((spaces,), domainind(T))))
    return BlockTensorMap(data, codomain, domain)
end

function BlockTensorMap(t::AbstractTensorMap)
    data = SparseArray{typeof(t), numind(t)}(undef, one.(1:numind(t)))
    data[1] = t
    return BlockTensorMap(data, SumSpace.(codomain(t)), SumSpace.(domain(t)))
end

function _extract_spaces(data::SparseArray{T,N}) where {T<:AbstractTensorMap,N}
    spaces = ntuple(i -> Vector{Union{spacetype(T),Missing}}(missing, size(data, i)), N)
    for (ind, val) in nonzero_pairs(data)
        for i in 1:N
            if ismissing(spaces[i][ind.I[i]])
                spaces[i][ind.I[i]] = space(val, i)
            else
                @assert spaces[i][ind.I[i]] == space(val, i) "incompatible spaces $(spaces[i][ind.I[i]]) and $(space(val, i))"
            end
        end
    end
    
    return map(spaces) do V
        @assert !any(ismissing.(V)) "cannot deduce spaces"
        return SumSpace(collect(skipmissing(V)))
    end
end

function _extract_spaces(data::AbstractArray{T,N}) where {T<:AbstractTensorMap,N}
    return map(1:N) do i
        return SumSpace(map(1:size(data, i)) do j
                            page = selectdim(data, i, j)
                            s = space(first(page), i)
                            @assert all(==(s), space.(page, i)) "incompatible spaces"
                            return s
                        end)
    end
end

function BlockTensorMap(::UndefInitializer, cod::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S′,S<:SumSpace{<:Any,S′}}
    T = tensormaptype(S′,N₁,N₂)
    data = SparseArray{T}(undef, ntuple(i -> i > N₁ ? length(dom[i-N₁]) : length(cod[i]), N₁ + N₂))
    return BlockTensorMap(data, cod, dom)
end


###################
## AbstractArray ##
###################
Base.parent(A::BlockTensorMap) = A.data
Base.eltype(A::BlockTensorMap) = eltype(parent(A))
Base.size(A::BlockTensorMap) = size(parent(A))
Base.getindex(A::BlockTensorMap, I::Int...) = getindex(parent(A), I...)
Base.setindex!(A::BlockTensorMap, v, I::Int...) = setindex!(parent(A), v, I...)

function Base.getindex(A::BlockTensorMap, I::CartesianIndex{N}) where {N}
    ndims(A) == N || throw(ArgumentError("invalid index style"))
    @boundscheck checkbounds(parent(A), I)
    return get(parent(A).data, I) do
        inds = Tuple(I)
        codomain = ProductSpace(getindex.(TensorKit.codomain(A), inds[domainind(A)])...)
        domain = ProductSpace(getindex.(TensorKit.domain(A), inds[codomainind(A)])...)
        return TensorMap(zeros, scalartype(A), codomain, domain)
    end
end

for f in (:nonzero_keys, :nonzero_length, :nonzero_pairs, :nonzero_values)
    @eval SparseArrayKit.$f(A::BlockTensorMap) = $f(parent(A))
end

### show and friends

function Base.show(io::IO, ::MIME"text/plain", x::BlockTensorMap)
    xnnz = nonzero_length(x)
    print(io, join(size(x), "×"), " ", typeof(x), " with ", xnnz, " stored ",
          xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
Base.show(io::IO, x::BlockTensorMap) = show(convert(IOContext, io), x)
function Base.show(io::IOContext, x::BlockTensorMap)
    nzind = nonzero_keys(x)
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    pads = map(1:ndims(x)) do i
        return ndigits(maximum(getindex.(nzind, i)))
    end
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for (k, (ind, val)) in enumerate(nonzero_pairs(x))
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', join(lpad.(Tuple(ind), pads), ","), "]  =  ", val)
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", join(" " .^ pads, " "), "   \u22ee")
        end
    end
end