# Linear Algebra
# --------------
Base.:(+)(t::BlockTensorMap, t2::BlockTensorMap) = add(t, t2)
Base.:(-)(t::BlockTensorMap, t2::BlockTensorMap) = add(t, t2, -one(scalartype(t)))
Base.:(*)(t::BlockTensorMap, α::Number) = scale(t, α)
Base.:(*)(α::Number, t::BlockTensorMap) = scale(t, α)
Base.:(/)(t::BlockTensorMap, α::Number) = scale(t, inv(α))
Base.:(\)(α::Number, t::BlockTensorMap) = scale(t, inv(α))

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

function TensorKit.compose_dest(A::AbstractBlockTensorMap, B::AbstractBlockTensorMap)
    T = Base.promote_op(LinearAlgebra.matprod, scalartype(A), scalartype(B))
    V = codomain(A) ← domain(B)
    return similar(issparse(A) ? B : A, T, V)
end

# function LinearAlgebra.mul!(
#     C::BlockTensorMap, A::BlockTensorMap, B::BlockTensorMap, α::Number, β::Number
# )
#     # @assert !(C isa SparseBlockTensorMap)
#     szA = size(A)
#     szB = size(B)
#     csizeA = TupleTools.getindices(szA, domainind(A))
#     csizeB = TupleTools.getindices(szB, codomainind(B))
#     osizeA = TupleTools.getindices(szA, codomainind(A))
#     osizeB = TupleTools.getindices(szB, domainind(B))
#
#     AS = sreshape(StridedView(parent(A)), prod(osizeA), prod(csizeA))
#     BS = sreshape(StridedView(parent(B)), prod(csizeB), prod(osizeB))
#     CS = sreshape(StridedView(parent(C)), prod(osizeA), prod(osizeB))
#
#     for i in axes(AS, 1), j in axes(BS, 2)
#         for k in axes(AS, 2)
#             if k == 1
#                 mul!(CS[i, j], AS[i, k], BS[k, j], α, β)
#             else
#                 mul!(CS[i, j], AS[i, k], BS[k, j], α, One())
#             end
#         end
#     end
#     return C
# end
# function LinearAlgebra.mul!(C::BlockTensorMap, A::BlockTensorMap, B::SparseBlockTensorMap,
#                             α::Number, β::Number)
#     @assert !(C isa SparseBlockTensorMap)
#     KB = collect(nonzero_keys(B))
#     for I in eachindex(IndexCartesian(), C)
#         allk = findall(KB) do J
#             return getindices(J.I, domainind(B)) == getindices(I.I, domainind(C))
#         end
#         if isempty(allk)
#             C[I] = scale!(C[I], β)
#             continue
#         end
#         for (ik, k) in enumerate(KB[allk])
#             IA = (getindices(I.I, codomainind(C))..., getindices(k.I, codomainind(B))...)
#             # IB = (getindices(k.I, domainind(B))..., getindices(I.I, domainind(C))...)
#             C[I] = mul!(C[I], A[IA...], B[k], α, ik == 1 ? β : One())
#         end
#     end
#     return C
# end
# function LinearAlgebra.mul!(C::BlockTensorMap, A::SparseBlockTensorMap, B::BlockTensorMap,
#                             α::Number, β::Number)
#     @assert !(C isa SparseBlockTensorMap)
#     KA = collect(nonzero_keys(A))
#     for I in eachindex(IndexCartesian(), C)
#         allk = findall(KA) do J
#             return getindices(J.I, codomainind(A)) == getindices(I.I, codomainind(C))
#         end
#         if isempty(allk)
#             C[I] = scale!(C[I], β)
#             continue
#         end
#         for (ik, k) in enumerate(KA[allk])
#             IB = (getindices(k.I, domainind(A))..., getindices(I.I, domainind(C))...)
#             C[I] = mul!(C[I], A[k], B[IB...], α, ik == 1 ? β : One())
#         end
#     end
#     return C
# end
# function LinearAlgebra.mul!(C::BlockTensorMap, A::SparseBlockTensorMap,
#                             B::SparseBlockTensorMap, α::Number, β::Number)
#     KA = collect(nonzero_keys(A))
#     KB = collect(nonzero_keys(B))
#     for I in eachindex(IndexCartesian(), C)
#         allka = findall(KA) do J
#             return getindices(J.I, codomainind(A)) == getindices(I.I, codomainind(C))
#         end
#         if isempty(allka)
#             C[I] = scale!(C[I], β)
#             continue
#         end
#
#         allkb = findall(KB) do J
#             return getindices(J.I, domainind(B)) == getindices(I.I, domainind(C))
#         end
#         if isempty(allkb)
#             C[I] = scale!(C[I], β)
#             continue
#         end
#
#         allk = CartesianIndex.(intersect(map(x -> getindices(x.I, domainind(A)),
#                                              KA[allka]),
#                                          map(x -> getindices(x.I, codomainind(B)),
#                                              KB[allkb])))
#         if isempty(allk)
#             C[I] = scale!(C[I], β)
#             continue
#         end
#         for (ik, k) in enumerate(allk)
#             IA = (getindices(I.I, codomainind(C))..., k.I...)
#             IB = (k.I..., getindices(I.I, domainind(C))...)
#             C[I] = mul!(C[I], A[IA...], B[IB...], α, ik == 1 ? β : One())
#         end
#     end
#     return C
# end
# function LinearAlgebra.mul!(
#     C::BlockTensorMap, A::AbstractTensorMap, B::BlockTensorMap, α::Number, β::Number
# )
#     # @assert !(C isa SparseBlockTensorMap)
#     szB = size(B)
#     csizeB = TupleTools.getindices(szB, codomainind(B))
#     osizeB = TupleTools.getindices(szB, domainind(B))
#
#     BS = sreshape(StridedView(parent(B)), prod(csizeB), prod(osizeB))
#     CS = sreshape(StridedView(parent(C)), 1, prod(osizeB))
#     for j in axes(BS, 2)
#         for k in axes(BS, 1)
#             if k == 1
#                 mul!(CS[1, j], A, BS[k, j], α, β)
#             else
#                 mul!(CS[1, j], A, BS[k, j], α, One())
#             end
#         end
#     end
#     return C
# end
# function LinearAlgebra.mul!(
#     C::BlockTensorMap, A::BlockTensorMap, B::AbstractTensorMap, α::Number, β::Number
# )
#     # @assert !(C isa SparseBlockTensorMap)
#     szA = size(A)
#     csizeA = TupleTools.getindices(szA, domainind(A))
#     osizeA = TupleTools.getindices(szA, codomainind(A))
#
#     AS = sreshape(StridedView(parent(A)), prod(osizeA), prod(csizeA))
#     CS = sreshape(StridedView(parent(C)), prod(osizeA), 1)
#     for i in axes(AS, 1)
#         for k in axes(AS, 2)
#             if k == 1
#                 mul!(CS[i, 1], AS[i, k], B, α, β)
#             else
#                 mul!(CS[i, 1], AS[i, k], B, α, One())
#             end
#         end
#     end
#     return C
# end
# function LinearAlgebra.mul!(
#     C::TensorMap, A::BlockTensorMap, B::BlockTensorMap, α::Number, β::Number
# )
#     C′ = convert(BlockTensorMap, C)
#     mul!(C′, A, B, α, β)
#     return C
# end
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

function LinearAlgebra.norm(tA::BlockTensorMap, p::Real=2)
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
