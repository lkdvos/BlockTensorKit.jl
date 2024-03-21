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

function LinearAlgebra.lmul!(α::Number, t::BlockTensorMap)
    for v in nonzero_values(t)
        lmul!(α, v)
    end
    return t
end

function LinearAlgebra.rmul!(t::BlockTensorMap, α::Number)
    for v in nonzero_values(t)
        rmul!(v, α)
    end
    return t
end

function LinearAlgebra.norm(tA::BlockTensorMap,
                            p::Real=2)
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
