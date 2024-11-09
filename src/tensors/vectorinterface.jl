# zerovector
# ----------
VI.zerovector!(t::AbstractBlockTensorMap) = (zerovector!(parent(t)); t)
VI.zerovector!(t::SparseBlockTensorMap) = (empty!(t.data); t)

function VI.scale!(t::AbstractBlockTensorMap, α::Number)
    foreach(Base.Fix2(scale!, α), nonzero_values(t))
    return t
end

function VI.scale!(ty::BlockTensorMap, tx::BlockTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    scale!(parent(ty), parent(tx), α)
    return ty
end
function VI.scale!(ty::SparseBlockTensorMap, tx::SparseBlockTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    # remove elements that are not in tx
    for k in setdiff(nonzero_keys(ty), nonzero_keys(tx))
        delete!(ty.data, k)
    end
    # in-place scale elements that are in both
    for k in intersect(nonzero_keys(ty), nonzero_keys(tx))
        ty[k] = scale!(ty[k], tx[k], α)
    end
    # new scale for elements in x that are not in y
    for k in setdiff(nonzero_keys(tx), nonzero_keys(ty))
        ty[k] = scale(tx[k], α)
    end
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
function VI.add!(ty::SparseBlockTensorMap, tx::SparseBlockTensorMap, α::Number, β::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    for (k, v) in nonzero_pairs(tx)
        ty[k] = α * ty[k] + β * v
    end
    return ty
end

# inner
# -----
function VI.inner(x::BlockTensorMap, y::BlockTensorMap)
    space(y) == space(x) || throw(SpaceMismatch())
    return inner(parent(x), parent(y))
end
function VI.inner(x::SparseBlockTensorMap, y::SparseBlockTensorMap)
    space(x) == space(y) || throw(SpaceMismatch())
    both_nonzero = intersect(nonzero_keys(x), nonzero_keys(y))
    T = VI.promote_inner(x, y)
    return sum(both_nonzero; init=zero(T)) do k
        inner(x[k], y[k])
    end
end
