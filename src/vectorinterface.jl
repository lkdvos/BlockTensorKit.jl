# VectorInterface
# ---------------

# function VI.zerovector(t::BlockTensorMap, ::Type{E}) where {E<:Number}
#     data = zerovector(parent(t), E)
#     return BlockTensorMap{E,spacetype(t),numout(t),numin(t),typeof(data)}(data, codomain(t), domain(t))
# end
VI.zerovector!(t::BlockTensorMap) = (zerovector!(parent(t)); t)
VI.zerovector!(t::SparseBlockTensorMap) = (empty!(t.data); t)
# VI.zerovector!(t::SparseBlockTensorMap) = (empty!(t.data.data); t)
# VI.zerovector!!(t::BlockTensorMap) = zerovector!(t)

# function VI.scale(t::BlockTensorMap, α::Number)
#     t′ = zerovector(t, VI.promote_scale(t, α))
#     scale!(t′, t, α)
#     return t′
# end
function VI.scale!(t::BlockTensorMap, α::Number)
    scale!(parent(t), α)
    return t
end
function VI.scale!(t::SparseBlockTensorMap, α::Number)
    if iszero(α)
        return zerovector!(t)
    else
        for v in nonzero_values(t)
            scale!(v, α)
        end
    end
    return t
end

function VI.scale!(ty::BlockTensorMap, tx::BlockTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    scale!(parent(ty), parent(tx), α)
    return ty
end

# function VI.scale!(ty::BlockTensorMap, tx::BlockTensorMap,
#                    α::Number)
#     space(ty) == space(tx) || throw(SpaceMismatch())
#     # delete all entries in ty that are not in tx
#     for I in setdiff(nonzero_keys(ty), nonzero_keys(tx))
#         delete!(ty.data, I)
#     end
#     # in-place scale elements from tx (getindex might allocate!)
#     for (I, v) in nonzero_pairs(tx)
#         ty[I] = scale!(ty[I], v, α)
#     end
#     return ty
# end
# function VI.scale!!(x::BlockTensorMap, α::Number)
#     return VI.promote_scale(x, α) <: scalartype(x) ? scale!(x, α) : scale(x, α)
# end
# function VI.scale!!(y::BlockTensorMap, x::BlockTensorMap,
#                     α::Number)
#     return VI.promote_scale(x, α) <: scalartype(y) ? scale!(y, x, α) : scale(x, α)
# end

# function VI.add(y::BlockTensorMap, x::BlockTensorMap, α::Number,
#                 β::Number)
#     space(y) == space(x) || throw(TK.SpaceMisMatch())
#     T = VI.promote_add(y, x, α, β)
#     z = zerovector(y, T)
#     # TODO: combine these operations where possible
#     scale!(z, y, β)
#     return add!(z, x, α)
# end
# function VI.add!(y::BlockTensorMap, x::BlockTensorMap, α::Number,
#                  β::Number)
#     space(y) == space(x) || throw(SpaceMisMatch())
#     # TODO: combine these operations where possible
#     scale!(y, β)
#     for (k, v) in nonzero_pairs(x)
#         y[k] = add!(y[k], v, α)
#     end
#     return y
# end
function VI.add!(ty::BlockTensorMap, tx::BlockTensorMap, α::Number, β::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    add!(parent(ty), parent(tx), α, β)
    return ty
end
# function VI.add!!(y::BlockTensorMap, x::BlockTensorMap, α::Number,
#                   β::Number)
#     return promote_add(y, x, α, β) <: scalartype(y) ? add!(y, x, α, β) : add(y, x, α, β)
# end

function VI.inner(x::BlockTensorMap, y::BlockTensorMap)
    space(y) == space(x) || throw(SpaceMismatch())
    return inner(parent(x), parent(y))
end

# VI.scalartype(::Type{<:BlockTensorMap{E}}) where {E} = E
