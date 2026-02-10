# zerovector
# ----------
VI.zerovector!(t::AbstractBlockTensorMap) = (zerovector!(parent(t)); t)
VI.zerovector!(t::SparseBlockTensorMap) = (empty!(t.data); t)

# scale
# -----
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
    y_notin_x = setdiff(nonzero_keys(ty), nonzero_keys(tx))
    x_notin_y = setdiff(nonzero_keys(tx), nonzero_keys(ty))
    inboth = intersect(nonzero_keys(ty), nonzero_keys(tx))

    # remove elements that are not in tx
    for k in y_notin_x
        delete!(ty.data, k)
    end
    # in-place scale elements that are in both
    for k in inboth
        ty[k] = scale!(ty[k], tx[k], α)
    end
    # new scale for elements in x that are not in y
    for k in x_notin_y
        ty[k] = scale(tx[k], α)
    end

    return ty
end

# add
# ---
function VI.add(ty::AbstractBlockTensorMap, tx::AbstractBlockTensorMap, α::Number, β::Number)
    S = TK.check_spacetype(ty, tx)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))

    # result type defaults to TensorMap if the types don't match to avoid assymmetric
    # implementation via zerovector(ty, T) vs zerovector(tx, T)
    # This would give issues for example with DiagonalTensorMap + TensorMap
    T = VectorInterface.promote_add(ty, tx, α, β)
    tdst = if typeof(ty) === typeof(tx)
        zerovector(ty, T)
    else
        M = TK.promote_storagetype(TK.similarstoragetype(ty, T), TK.similarstoragetype(tx, T))
        if issparse(ty) && issparse(tx)
            sparseblocktensormaptype(S, numout(ty), numin(ty), M)(undef, space(ty))
        else
            blocktensormaptype(S, numout(ty), numin(ty), M)(undef, space(ty))
        end
    end

    return add!(scale!(tdst, ty, β), tx, α)
end

function VI.add!(ty::BlockTensorMap, tx::BlockTensorMap, α::Number, β::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    add!(parent(ty), parent(tx), α, β)
    return ty
end
function VI.add!(ty::SparseBlockTensorMap, tx::SparseBlockTensorMap, α::Number, β::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    y_notin_x = setdiff(nonzero_keys(ty), nonzero_keys(tx))
    x_notin_y = setdiff(nonzero_keys(tx), nonzero_keys(ty))
    inboth = intersect(nonzero_keys(ty), nonzero_keys(tx))

    for k in y_notin_x
        ty[k] = scale!!(ty[k], β)
    end
    for k in x_notin_y
        ty[k] = scale(tx[k], α)
    end
    for k in inboth
        ty[k] = add!!(ty[k], tx[k], α, β)
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
    return sum(both_nonzero; init = zero(T)) do k
        inner(x[k], y[k])
    end
end
