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
        copyto!(tdst[f₁, f₂], t[f₁, f₂])
    end

    return tdst
end

function Base.convert(
    ::Type{TT}, t::AbstractBlockTensorMap
) where {TT<:AbstractBlockTensorMap}
    t isa TT && return t
    tdst = similar(TT, space(t))
    for (I, v) in nonzero_pairs(t)
        tdst[I] = v
    end
    return tdst
end

TensorKit.TensorMap(t::AbstractBlockTensorMap) = convert(TensorMap, t)