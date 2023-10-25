# TensorOperations Interface: BlockTensorMaps
# -------------------------------------------
TO.tensorstructure(t::BlockTensorMap) = space(t)
function TO.tensorstructure(t::BlockTensorMap, i::Int, conjA::Symbol)
    return conjA == :N ? space(t, i) : conj(space(t, i))
end

function TO.tensorscalar(t::BlockTensorMap)
    return ndims(t) == 0 ? tensorscalar(t[]) : throw(DimensionMismatch())
end

function TO.tensoradd!(C::BlockTensorMap{S}, A::BlockTensorMap{S}, pA::Index2Tuple,
                       conjA::Symbol, α::Number, β::Number, backend::Backend...) where {S}
    argcheck_tensoradd(parent(C), parent(A), pA)
    dimcheck_tensoradd(parent(C), parent(A), pA)

    let indCinA = linearize(pA)
        for iA in CartesianIndices(A)
            iC = CartesianIndex(getindices(iA.I, indCinA))
            C[iC] = tensoradd!(C[iC], A[iA], pA, conjA, α, β, backend...)
        end
    end

    return C
end

function TO.tensorcontract!(C::BlockTensorMap{S}, pAB::Index2Tuple,
                            A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::Backend...) where {S}
    argcheck_tensorcontract(parent(C), pAB, parent(A), pA, parent(B), pB)
    dimcheck_tensorcontract(parent(C), pAB, parent(A), pA, parent(B), pB)
    
    for (i, j) in zip(pA[2], pB[1])
        TO.checkcontractible(A, i, conjA, B, j, conjB, "$i and $j")
    end
    V = (TO.tensorstructure.(Ref(A), pA[1], conjA)..., TO.tensorstructure.(Ref(B), pB[2], conjB)...)
    for (i, j) in enumerate(linearize(pAB))
        V[j] == space(C, i) || throw(SpaceMismatch("incompatible spaces for $i: $(V[j]) ≠ $(space(C, i))"))
    end
    
    
    
    sizeA = size(A)
    sizeB = size(B)
    csizeA = getindices(sizeA, pA[2])
    csizeB = getindices(sizeB, pB[1])
    osizeA = getindices(sizeA, pA[1])
    osizeB = getindices(sizeB, pB[2])

    scale!(C, β)
    AS = sreshape(permutedims(StridedView(parent(A)), linearize(pA)),
                  (osizeA..., one.(osizeB)..., csizeA...))
    BS = sreshape(permutedims(StridedView(parent(B)), linearize(reverse(pB))),
                  (one.(osizeA)..., osizeB..., csizeB...))
    CS = sreshape(permutedims(StridedView(parent(C)), TupleTools.invperm(linearize(pAB))),
                  (osizeA..., osizeB..., one.(csizeA)...))
    tensorcontract!.(CS, Ref(pAB), AS, Ref(pA), conjA, BS, Ref(pB), conjB, α, One())

    return C
end

function TO.tensortrace!(C::BlockTensorMap{S}, pC::Index2Tuple,
                                       A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                       α::Number, β::Number) where {S}
    argcheck_tensortrace(parent(C), pC, parent(A), pA)
    dimcheck_tensortrace(parent(C), pC, parent(A), pA)


    scale!(C, β)
    for (IA, v) in nonzero_pairs(A)
        IAc1 = CartesianIndex(getindices(IA.I, pA[1]))
        IAc2 = CartesianIndex(getindices(IA.I, pA[2]))
        IAc1 == IAc2 || continue
        IC = CartesianIndex(getindices(IA.I, linearize(pC)))
        C[IC] = tensortrace!(C[IC], pC, v, pA, conjA, α, One())
    end
    return C
end

function TO.tensoralloc(::Type{BlockTensorMap{S,N₁,N₂,A}},
                        structure::TensorMapSpace{SumSpace{S},N₁,N₂}, istemp=false,
                        backend::Backend...) where {S,N₁,N₂,A}
    return BlockTensorMap(undef, A, structure)
end

function TO.tensorfree!(t::BlockTensorMap, backend::Backend...)
    for t in parent(t)
        TO.tensorfree!(t, backend...)
    end
    return nothing
end

function TO.tensoradd_type(TC::Type{<:Number}, pC::Index2Tuple{N₁,N₂},
                           ::BlockTensorMap{S}, conjA::Symbol) where {S,N₁,N₂}
    T = tensormaptype(S, N₁, N₂, TC)
    return BlockTensorMap{S,N₁,N₂,Array{T,(N₁ + N₂)}}
end

function TO.tensoradd_structure(pC::Index2Tuple{N₁,N₂}, A::BlockTensorMap{S},
                                conjA::Symbol) where {S,N₁,N₂}
    V = conjA == :N ? space(A) : space(A)'
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(V), pC[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(V), pC[2])))
    return dom → cod
end

function TO.tensorcontract_type(TC::Type{<:Number}, pC::Index2Tuple{N₁,N₂},
                                A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                                istemp=false, backend::Backend...) where {S,N₁,N₂}
    M = TensorKit.similarstoragetype(A, TC)
    M == TensorKit.similarstoragetype(B, TC) ||
        throw(ArgumentError("incompatible storage types"))
    T = tensormaptype(S, N₁, N₂, M)
    return BlockTensorMap{S,N₁,N₂,Array{T,(N₁ + N₂)}}
end

function TO.tensorcontract_structure(pC::Index2Tuple{N₁,N₂}, A::BlockTensorMap{S},
                                     pA::Index2Tuple, conjA::Symbol, B::BlockTensorMap{S},
                                     pB::Index2Tuple, conjB::Symbol) where {S,N₁,N₂}
    spaces1 = TO.flag2op(conjA).(space.(Ref(A), pA[1]))
    spaces2 = TO.flag2op(conjB).(space.(Ref(B), pB[2]))
    spaces = (spaces1..., spaces2...)
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(spaces), pC[2])))
    
    return dom → cod
end

function TO.checkcontractible(tA::BlockTensorMap{S}, iA::Int, conjA::Symbol,
                              tB::BlockTensorMap{S}, iB::Int, conjB::Symbol,
                              label) where {S}
    sA = TO.tensorstructure(tA, iA, conjA)'
    sB = TO.tensorstructure(tB, iB, conjB)
    sA == sB ||
        throw(SpaceMismatch("incompatible spaces for $label: $sA ≠ $sB"))
    return nothing
end

# TensorOperations interface: BlockTensorMaps - TensorMaps
# ---------------------------------------------------------

# function TensorOperations.tensorcontract!(
#     C::BlockTensorMap{S},
#     pC::Index2Tuple,
#     A::AbstractTensorMap{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::AbstractTensorMap{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     α::Number,
#     β::Number,
# ) where {S}
#     blockA = convert(BlockTensorMap, A)
#     blockB = convert(BlockTensorMap, B)
#     return tensorcontract!(C, pC, blockA, pA, conjA, blockB, pB, conjB, α, β)
# end

# function TensorOperations.tensorcontract!(
#     C::TensorMap{S},
#     pC::Index2Tuple,
#     A::BlockTensorMap{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::AbstractTensorMap{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     α::Number,
#     β::Number,
# ) where {S}
#     blockB = convert(BlockTensorMap, B)
#     return tensorcontract!(C, pC, A, pA, conjA, blockB, pB, conjB, α, β)
# end
# function TensorOperations.tensorcontract!(
#     C::TensorMap{S},
#     pC::Index2Tuple,
#     A::AbstractTensorMap{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::BlockTensorMap{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     α::Number,
#     β::Number,
# ) where {S}
#     blockA = convert(BlockTensorMap, A)
#     return tensorcontract!(C, pC, A, pA, conjA, blockB, pB, conjB, α, β)
# end
# function TensorOperations.tensorcontract!(
#     C::TensorMap{S},
#     pC::Index2Tuple,
#     A::BlockTensorMap{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::BlockTensorMap{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     α::Number,
#     β::Number,
# ) where {S}
#     C′ = convert(BlockTensorMap, C)
#     tensorcontract!(C′, pC, A, pA, conjA, B, pB, conjB, α, β)
#     return C
# end
