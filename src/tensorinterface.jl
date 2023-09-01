function TensorOperations.tensorscalar(t::BlockTensorMap)
    return ndims(t) == 0 ? tensorscalar(t[]) : throw(DimensionMismatch())
end

function TensorOperations.tensoradd!(C::BlockTensorMap{S},
                       A::BlockTensorMap{S}, pA::Index2Tuple,
                       conjA::Symbol, α::Number, β::Number) where {S}
    ndims(C) == ndims(A) == sum(length.(pA)) ||
        throw(IndexError("Invalid permutation of length $N: $pA"))
    size(C) == getindices(size(A), linearize(pA)) ||
        throw(DimensionMismatch("non-matching sizes while adding arrays"))

    let indCinA = linearize(pA)
        for iA in CartesianIndices(A)
            iC = CartesianIndex(getindices(iA.I, indCinA))
            C[iC] = tensoradd!(C[iC], A[iA], pA, conjA, α, β)
        end
    end
    
    return C
end

function TensorOperations.tensorcontract!(C::BlockTensorMap{S}, pC::Index2Tuple,
                            A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number) where {S}
    (sum(length.(pA)) == ndims(A) && isperm(linearize(pA))) ||
        throw(IndexError("invalid permutation of length $(ndims(A)): $pA"))
    (sum(length.(pB)) == ndims(B) && isperm(linearize(pB))) ||
        throw(IndexError("invalid permutation of length $(ndims(B)): $pB"))
    (length(pA[1]) + length(pB[2]) == ndims(C)) ||
        throw(IndexError("non-matching output indices in contraction"))
    (ndims(C) == length(linearize(pC)) && isperm(linearize(pC))) ||
        throw(IndexError("invalid permutation of length $(ndims(C)): $pC"))

    sizeA = size(A)
    sizeB = size(B)
    sizeC = size(C)

    csizeA = getindices(sizeA, pA[2])
    csizeB = getindices(sizeB, pB[1])
    osizeA = getindices(sizeA, pA[1])
    osizeB = getindices(sizeB, pB[2])
    csizeA == csizeB ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    getindices((osizeA..., osizeB...), linearize(pC)) == sizeC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))

    AS = sreshape(permutedims(StridedView(parent(A)), linearize(pA)),
                  (osizeA..., one.(osizeB)..., csizeA...))
    BS = sreshape(permutedims(StridedView(parent(B)), linearize(reverse(pB))),
                  (one.(osizeA)..., osizeB..., csizeB...))
    CS = sreshape(permutedims(StridedView(parent(C)), TupleTools.invperm(linearize(pC))),
                  (osizeA..., osizeB..., one.(csizeA)...))

    isone(β) || rmul!(CS, β)

    tensorcontract!.(CS, Ref(pC), AS, Ref(pA), conjA, BS, Ref(pB), conjB, α, true)
    return C
end

function TensorOperations.tensortrace!(C::BlockTensorMap{S}, pC::Index2Tuple,
                                       A::BlockTensorMap{S},
                         pA::Index2Tuple,
                         conjA::Symbol, α::Number, β::Number) where {S}
    NA, NC = ndims(A), ndims(C)
    NC == sum(length.(pC)) ||
        throw(IndexError("Invalid selection of $NC out of $NA: $pC"))
    NA - NC == 2 * length(pA[1]) == 2 * length(pA[2]) ||
        throw(IndexError("invalid number of trace dimension"))
    pA′ = (linearize(pC)..., linearize(pA)...)
    isperm(pA′) ||
        throw(IndexError("invalid permutation of length $NA: $(pA′)"))

    sizeA = size(A)
    sizeC = size(C)

    getindices(sizeA, pA[1]) == getindices(sizeA, pA[2]) ||
        throw(DimensionMismatch("non-matching trace sizes"))
    sizeC == getindices(sizeA, linearize(pC)) ||
        throw(DimensionMismatch("non-matching sizes"))

    β == one(β) || LinearAlgebra.lmul!(β, C)
    for (IA, v) in nonzero_pairs(A)
        IAc1 = CartesianIndex(getindices(IA.I, pA[1]))
        IAc2 = CartesianIndex(getindices(IA.I, pA[2]))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(getindices(IA.I, linearize(pC)))
        C[IC] = tensortrace!(C[IC], pC, v, pA, conjA, α, one(β))
    end
    return C
end