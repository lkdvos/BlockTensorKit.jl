# TensorOperations Interface: BlockTensorMaps
# -------------------------------------------
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

function TO.tensorcontract!(C::BlockTensorMap{S}, pC::Index2Tuple,
                            A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::Backend...) where {S}
    argcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)
    dimcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)

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
    CS = sreshape(permutedims(StridedView(parent(C)), TupleTools.invperm(linearize(pC))),
                  (osizeA..., osizeB..., one.(csizeA)...))
    tensorcontract!.(CS, Ref(pC), AS, Ref(pA), conjA, BS, Ref(pB), conjB, α, One())

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
                        structure::TensorKit.HomSpace{SumSpace{S},N₁,N₂}, istemp=false,
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
    VA = conjA == :N ? space(A) : space(A)'
    VB = conjB == :N ? space(B) : space(B)'
    V = (getindex.(Ref(VA), pA[1])..., getindex.(Ref(VB), pB[2])...)
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(V), pC[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(V), pC[2])))
    return dom → cod
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
