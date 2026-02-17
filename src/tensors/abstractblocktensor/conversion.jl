# Conversion
# ----------

function _copy_subblocks!(tdst, tsrc)
    S = spacetype(tsrc)
    N₁, N₂ = numout(tsrc), numin(tsrc)
    for ((f₁, f₂), arr) in subblocks(tdst)
        blockax = ntuple(N₁ + N₂) do i
            return if i <= N₁
                blockedrange(map(Base.Fix2(dim, f₁.uncoupled[i]), space(tsrc, i)))
            else
                blockedrange(map(Base.Fix2(dim, f₂.uncoupled[i - N₁]), space(tsrc, i)'))
            end
        end

        for (k, v) in nonzero_pairs(tsrc)
            indices = getindex.(blockax, Block.(Tuple(k)))
            arr_slice = arr[indices...]
            # need to check for empty since fusion tree pair might not be present
            isempty(arr_slice) || copy!(arr_slice, v[f₁, f₂])
        end
    end
    return tdst
end

function Base.convert(::Type{TensorMap}, t::AbstractBlockTensorMap)
    S = spacetype(t)
    N₁, N₂ = numout(t), numin(t)
    cod = ProductSpace{S, N₁}(oplus.(codomain(t).spaces))
    dom = ProductSpace{S, N₂}(oplus.(domain(t).spaces))
    tdst = TensorKit.TensorMapWithStorage{scalartype(t), storagetype(t)}(undef, cod, dom)

    issparse(t) && zerovector!(tdst)
    _copy_subblocks!(tdst, t)
    return tdst
end

function Base.convert(::Type{TT}, t::AbstractBlockTensorMap) where {TT <: TensorKit.TensorMap}
    S = spacetype(t)
    N₁, N₂ = numout(t), numin(t)
    cod = ProductSpace{S, N₁}(oplus.(codomain(t).spaces))
    dom = ProductSpace{S, N₂}(oplus.(domain(t).spaces))
    tdst = TT(undef, cod ← dom)
    issparse(t) && zerovector!(tdst)

    _copy_subblocks!(tdst, t)
    return tdst
end

function Base.convert(::Type{TT}, t::AbstractTensorMap) where {TT <: AbstractBlockTensorMap}
    t isa TT && return t
    if t isa AbstractBlockTensorMap
        tdst = similar(TT, space(t))
        for (I, v) in nonzero_pairs(t)
            tdst[I] = v
        end
    else
        S = spacetype(t)
        tdst = TT(
            undef,
            convert(ProductSumSpace{S, numout(t)}, codomain(t)),
            convert(ProductSumSpace{S, numin(t)}, domain(t)),
        )
        tdst[1] = t
    end
    return tdst
end

TensorKit.TensorMap(t::AbstractBlockTensorMap) = convert(TensorMap, t)
