# VectorInterface
# ---------------

function VI.zerovector(t::BlockTensorMap{E₁}, ::Type{E₂}) where {E₁,E₂<:Number}
    return similar(t, E₂, space(t))
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
        println(ty[I])
        ty[I] = scale!(ty[I], v, α)
    end
    return ty
end
function VI.scale!!(x::BlockTensorMap, α::Number)
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

VI.scalartype(::BlockTensorMap{E,S,N₁,N₂,N}) where {E,S,N₁,N₂,N} = E
