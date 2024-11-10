# Linear Algebra
# --------------
Base.:(+)(t::AbstractBlockTensorMap, t2::AbstractBlockTensorMap) = add(t, t2)
function Base.:(-)(t::AbstractBlockTensorMap, t2::AbstractBlockTensorMap)
    return add(t, t2, -one(scalartype(t)))
end
Base.:(*)(t::AbstractBlockTensorMap, α::Number) = scale(t, α)
Base.:(*)(α::Number, t::AbstractBlockTensorMap) = scale(t, α)
Base.:(/)(t::AbstractBlockTensorMap, α::Number) = scale(t, inv(α))
Base.:(\)(α::Number, t::AbstractBlockTensorMap) = scale(t, inv(α))

function LinearAlgebra.axpy!(α::Number, t1::BlockTensorMap, t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    for (i, v) in nonzero_pairs(t1)
        t2[i] = axpy!(α, v, t2[i])
    end
    return t2
end

function LinearAlgebra.axpby!(α::Number, t1::BlockTensorMap, β::Number, t2::BlockTensorMap)
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

# This is a generic implementation of `mul!` for BlockTensors that is used to make it easier
# to work with abstract element types, that might not support in-place operations.
# For now, the implementation might not be hyper-optimized, but the assumption is that we
# are in the limit where multiplying the individual elements is the bottleneck.
# With that in mind, we simply write the multiplication in terms of sparse tensors.
function LinearAlgebra.mul!(
    C::AbstractBlockTensorMap,
    A::AbstractBlockTensorMap,
    B::AbstractBlockTensorMap,
    α::Number,
    β::Number,
)
    compose(space(A), space(B)) == space(C) ||
        throw(SpaceMismatch(lazy"$(space(C)) ≠ $(space(A)) * $(space(B))"))

    Caxes = Base.Fix1(axes, C)
    Aaxes = Base.Fix1(axes, A)

    for I in Iterators.product(map(Caxes, codomainind(C))...),
        J in Iterators.product(map(Caxes, domainind(C))...)

        did_mul = false
        for K in Iterators.product(map(Aaxes, domainind(A))...)
            vA = get(A, CartesianIndex(I..., K...), nothing)
            isnothing(vA) && continue
            vB = get(B, CartesianIndex(K..., J...), nothing)
            isnothing(vB) && continue

            vC = get(C, CartesianIndex(I..., J...), nothing)
            if did_mul
                C[I..., J...] = _mul!!(vC, vA, vB, α, One())
            else
                C[I..., J...] = _mul!!(vC, vA, vB, α, β)
                did_mul = true
            end
        end
        # handle `β`
        if !did_mul
            vC = get(C, CartesianIndex(I..., J...), nothing)
            if !isnothing(vC)
                C[I..., J...] = scale!!(vC, β)
            end
        end
    end

    return C
end

_mull!!(::Nothing, A, B, α::Number, β::Number) = scale!!(A * B, α)
_mul!!(C, A, B, α::Number, β::Number) = add!!(C, A * B, α, β)
const _TM_CAN_MUL = Union{TensorMap,AdjointTensorMap{<:TensorMap}}
function _mul!!(C::_TM_CAN_MUL, A::_TM_CAN_MUL, B::_TM_CAN_MUL, α::Number, β::Number)
    return mul!(C, A, B, α, β)
end
# TODO: optimize other implementations

# ensure that mixes with AbstractBlockTensorMap and AbstractTensorMap behave as expected:
for (TC, TA, TB) in Iterators.product(
    Iterators.repeated((:AbstractTensorMap, :AbstractBlockTensorMap), 3)...
)
    (
        :AbstractBlockTensorMap ∉ (TC, TA, TB) ||
        all(==(:AbstractBlockTensorMap), (TC, TA, TB))
    ) && continue
    @eval function LinearAlgebra.mul!(C::$TC, A::$TA, B::$TB, α::Number, β::Number)
        A′ = A isa AbstractBlockTensorMap ? A : convert(BlockTensorMap, A)
        B′ = B isa AbstractBlockTensorMap ? B : convert(BlockTensorMap, B)
        if C isa AbstractBlockTensorMap
            return mul!(C, A′, B′, α, β)
        else
            C′ = convert(BlockTensorMap, C)
            C′ = mul!(C′, A′, B′, α, β)
            return only(C′)
        end
    end
end

function LinearAlgebra.norm(tA::BlockTensorMap, p::Real=2)
    vals = nonzero_values(tA)
    isempty(vals) && return norm(zero(scalartype(tA)), p)
    return LinearAlgebra.norm(norm.(vals), p)
end

for f in (:real, :imag)
    @eval function Base.$f(t::AbstractBlockTensorMap)
        if isreal(sectortype(spacetype(t)))
            TT = Base.promote_op($f, eltype(t))
            t′ = similar(t, TT)
            @inbounds for (k, v) in nonzero_pairs(t)
                t′[k] = $f(v)
            end
            return t′
        else
            msg = "`$f` has not been implemented for `BlockTensorMap{$(S)}`."
            throw(ArgumentError(msg))
        end
    end
end
