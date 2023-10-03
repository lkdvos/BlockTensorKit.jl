const SparseBlockTensorMap{S,N₁,N₂,T,N} = BlockTensorMap{S,N₁,N₂,
                                                         SparseArray{T,N}} where {T,N}

# function Base.getindex(t::SparseBlockTensorMap, I::Int...)
#     @assert length(I) == numind(t)
#     return getindex(t, CartesianIndex(I))
# end
# function Base.getindex(t::SparseBlockTensorMap, I::CartesianIndex)
#     @boundscheck checkbounds(t, I)
#     return get(parent(t).data, I) do
#         return TensorMap(zeros, scalartype(t), getsubspace(space(t), I))
#     end
# end

#===========================================================================================
    Linear Algebra
===========================================================================================#

function LinearAlgebra.axpy!(α::Number, t1::SparseBlockTensorMap, t2::SparseBlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    for (i, v) in nonzero_pairs(t1)
        t2[i] = axpy!(α, v, t2[i])
    end
    return t2
end

function LinearAlgebra.axpby!(α::Number, t1::SparseBlockTensorMap, β::Number,
                              t2::SparseBlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    rmul!(t2, β)
    for (i, v) in nonzero_pairs(t1)
        t2[i] = axpy!(α, v, t2[i])
    end
    return t2
end

function LinearAlgebra.dot(t1::SparseBlockTensorMap, t2::SparseBlockTensorMap)
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

function LinearAlgebra.mul!(C::SparseBlockTensorMap, α::Number, A::SparseBlockTensorMap)
    space(C) == space(A) || throw(SpaceMismatch())
    SparseArrayKit._zero!(parent(C))
    for (i, v) in nonzero_pairs(A)
        C[i] = mul!(C[i], α, v)
    end
    return C
end

function LinearAlgebra.norm(tA::SparseBlockTensorMap{S,N1,N2,A},
                            p::Real=2) where {S,N1,N2,A}
    vals = nonzero_values(tA)
    isempty(vals) && return norm(zero(scalartype(tA)), p)
    return LinearAlgebra.norm(norm.(vals), p)
end

function Base.real(t::SparseBlockTensorMap)
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

function Base.imag(t::SparseBlockTensorMap)
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

#===========================================================================================
    TensorOperations
===========================================================================================#

function TensorOperations.tensoradd!(C::SparseBlockTensorMap{S},
                                     A::SparseBlockTensorMap{S}, pA::Index2Tuple,
                                     conjA::Symbol, α::Number, β::Number) where {S}
    ndims(C) == ndims(A) == sum(length.(pA)) ||
        throw(IndexError("Invalid permutation of length $N: $pA"))
    size(C) == getindices(size(A), linearize(pA)) ||
        throw(DimensionMismatch("non-matching sizes while adding arrays"))

    let indCinA = linearize(pA)
        for (iA, vA) in nonzero_pairs(A)
            iC = CartesianIndex(getindices(iA.I, indCinA))
            C[iC] = tensoradd!(C[iC], vA, pA, conjA, α, β)
        end
    end
    return C
end

function TensorOperations.tensorcontract!(C::BlockTensorMap{S}, pC::Index2Tuple,
                                          A::BlockTensorMap{S}, pA::Index2Tuple,
                                          conjA::Symbol,
                                          B::SparseBlockTensorMap{S}, pB::Index2Tuple,
                                          conjB::Symbol,
                                          α::Number, β::Number) where {S}
    TensorOperations.argcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)
    TensorOperations.dimcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)

    sizeA = size(A)
    sizeB = size(B)

    scale!(C, β)

    keysA = sort!(collect(vec(keys(A))); by=IA -> CartesianIndex(getindices(IA.I, pA[2])))
    keysB = sort!(collect(nonzero_keys(B));
                  by=IB -> CartesianIndex(getindices(IB.I, pB[1])))

    iA = iB = 1
    @inbounds while iA <= length(keysA) && iB <= length(keysB)
        IA = keysA[iA]
        IB = keysB[iB]
        IAc = CartesianIndex(getindices(IA.I, pA[2]))
        IBc = CartesianIndex(getindices(IB.I, pB[1]))
        if IAc == IBc
            Ic = IAc
            jA = iA
            while jA < length(keysA)
                if CartesianIndex(getindices(keysA[jA + 1].I, pA[2])) == Ic
                    jA += 1
                else
                    break
                end
            end
            jB = iB
            while jB < length(keysB)
                if CartesianIndex(getindices(keysB[jB + 1].I, pB[1])) == Ic
                    jB += 1
                else
                    break
                end
            end
            rA = iA:jA
            rB = iB:jB
            if length(rA) < length(rB)
                for kB in rB
                    IB = keysB[kB]
                    IBo = CartesianIndex(getindices(IB.I, pB[2]))
                    vB = B[IB]
                    for kA in rA
                        IA = keysA[kA]
                        IAo = CartesianIndex(getindices(IA.I, pA[1]))
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(getindices(IABo.I, linearize(pC)))
                        vA = A[IA]
                        C[IC] = tensorcontract!(C[IC], pC, vA, pA, conjA, vB, pB, conjB, α,
                                                One())
                    end
                end
            else
                for kA in rA
                    IA = keysA[kA]
                    IAo = CartesianIndex(getindices(IA.I, pA[1]))
                    vA = A[IA]
                    for kB in rB
                        IB = keysB[kB]
                        IBo = CartesianIndex(getindices(IB.I, pB[2]))
                        vB = parent(B).data[IB]
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(getindices(IABo.I, linearize(pC)))
                        C[IC] = tensorcontract!(C[IC], pC, vA, pA, conjA, vB, pB, conjB, α,
                                                One())
                    end
                end
            end
            iA = jA + 1
            iB = jB + 1
        elseif IAc < IBc
            iA += 1
        else
            iB += 1
        end
    end

    return C
end

function TensorOperations.tensortrace!(C::SparseBlockTensorMap{S}, pC::Index2Tuple,
                                       A::SparseBlockTensorMap{S},
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

function Base.adjoint(t::BlockTensorMap{S,N₁,N₂,A}) where {S,N₁,N₂,A<:SparseArray}
    cod = domain(t)
    dom = codomain(t)

    sz_t = size(t)
    adjoint_inds = [domainind(t)..., codomainind(t)...]
    sz_t′ = sz_t[adjoint_inds]

    T = typeof(adjoint(first(nonzero_values(t))))

    data = SparseArray{T,N₁ + N₂}(undef, sz_t′)
    for (I, v) in nonzero_pairs(t)
        I′ = CartesianIndex(getindex.(Ref(I), adjoint_inds)...)
        v′ = adjoint(v)
        data[I′] = v′
    end

    return BlockTensorMap(data, cod, dom)
end

# function TensorKit.tensormap(f, cod::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:SumSpace,N₁,N₂}
#     sz = ntuple(i -> i > N₁ ? length(dom[i - N₁]) : length(cod[i]), N₁ + N₂)
#     data = map(CartesianIndices(sz)) do idx
#         subcod = getsubspace(cod, CartesianIndex(idx[1:N₁]...))
#         subdom = getsubspace(dom, CartesianIndex(idx[N₁+1:end]...))
#         tensormap(f, subcod, subdom)
#     end
#     return BlockTensorMap(data, cod, dom)
# end

# function TensorKit.tensormap(f, T::Type{<:Number}, cod::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:SumSpace,N₁,N₂}
#     sz = ntuple(i -> i > N₁ ? length(dom[i - N₁]) : length(cod[i]), N₁ + N₂)
#     data = map(CartesianIndices(sz)) do idx
#         subcod = getsubspace(cod, CartesianIndex(idx[1:N₁]...))
#         subdom = getsubspace(dom, CartesianIndex(idx[N₁+1:end]...))
#         tensormap(f, T, subcod, subdom)
#     end
#     return BlockTensorMap(data, cod, dom)
# end

Base.haskey(t::SparseBlockTensorMap, I::CartesianIndex) = haskey(parent(t).data, I)
