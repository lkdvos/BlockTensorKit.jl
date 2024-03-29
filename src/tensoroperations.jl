# TensorOperations
# ----------------

function TO.tensoradd_type(TC, ::Index2Tuple{N₁,N₂}, ::BlockTensorMap{S},
                           conjA::Symbol) where {S,N₁,N₂}
    T = tensormaptype(S, N₁, N₂, TC)
    return BlockTensorMap{S,N₁,N₂,T,N₁ + N₂}
end

function TO.tensoradd_structure(pC::Index2Tuple{N₁,N₂}, A::BlockTensorMap{S},
                                conjA::Symbol) where {S,N₁,N₂}
    if conjA == :N
        pC′ = pC
        V = space(A)
    else
        pC′ = TK.adjointtensorindices(A, pC)
        V = space(A)'
    end
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(V), pC′[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(V), pC′[2])))
    return dom → cod
end

function TO.tensorcontract_type(TC::Type{<:Number}, ::Index2Tuple{N₁,N₂},
                                A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                                istemp=false, backend::Backend...) where {S,N₁,N₂}
    M = TK.similarstoragetype(A, TC)
    M == TK.similarstoragetype(B, TC) ||
        throw(ArgumentError("incompatible storage types"))
    T = tensormaptype(S, N₁, N₂, M)
    return BlockTensorMap{S,N₁,N₂,T,N₁ + N₂}
end

# By default, make "dense" allocations
function TO.tensoralloc(::Type{BlockTensorMap{S,N1,N2,T,N}}, structure::TensorMapSumSpace,
                        istemp::Bool, backend::B...) where {S,N1,N2,T,N,B<:Backend}
    C = BlockTensorMap{S,N1,N2,T,N}(undef, structure)
    for I in eachindex(C)
        C[I] = TO.tensoralloc(T, getsubspace(structure, I), istemp, backend...)
    end
    return C
end

function TO.tensorfree!(t::BlockTensorMap, backend::Backend...)
    for v in nonzero_values(t)
        TO.tensorfree!(v, backend...)
    end
    return nothing
end

function TO.tensoradd!(C::BlockTensorMap{S}, pC::Index2Tuple,
                       A::BlockTensorMap{S}, conjA::Symbol,
                       α::Number, β::Number, backend::Backend...) where {S}
    argcheck_tensoradd(C, pC, A)
    dimcheck_tensoradd(C, pC, A)

    scale!(C, β)
    indCinA = linearize(pC)
    for (IA, v) in nonzero_pairs(A)
        IC = CartesianIndex(TupleTools.getindices(IA.I, indCinA))
        C[IC] = tensoradd!(C[IC], pC, v, conjA, α, One(), backend...)
    end
    return C
end

function TO.tensorcontract!(C::BlockTensorMap{S}, pC::Index2Tuple,
                            A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::BlockTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::Backend...) where {S}
    argcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)
    dimcheck_tensorcontract(parent(C), pC, parent(A), pA, parent(B), pB)

    scale!(C, β)

    keysA = sort!(collect(nonzero_keys(A));
                  by=IA -> CartesianIndex(getindices(IA.I, pA[2])))
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
                                                One(), backend...)
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
                                                One(), backend...)
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

function TO.tensortrace!(C::BlockTensorMap{S}, pC::Index2Tuple,
                         A::BlockTensorMap{S},
                         pA::Index2Tuple,
                         conjA::Symbol, α::Number, β::Number, backend::Backend...) where {S}
    argcheck_tensortrace(C, pC, A, pA)
    dimcheck_tensortrace(C, pC, A, pA)

    scale!(C, β)

    for (IA, v) in nonzero_pairs(A)
        IAc1 = CartesianIndex(getindices(IA.I, pA[1]))
        IAc2 = CartesianIndex(getindices(IA.I, pA[2]))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(getindices(IA.I, linearize(pC)))
        C[IC] = tensortrace!(C[IC], pC, v, pA, conjA, α, One(), backend...)
    end
    return C
end

function TO.tensorscalar(C::BlockTensorArray{T,0}) where {T}
    return isempty(C.data) ? zero(scalartype(C)) : tensorscalar(C[])
end

TO.tensorstructure(t::BlockTensorMap) = space(t)
function TO.tensorstructure(t::BlockTensorMap, iA::Int, conjA::Symbol)
    return conjA == :N ? space(t, iA) : conj(space(t, iA))
end

function TO.tensorcontract_structure(pC::Index2Tuple{N₁,N₂},
                                     A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                     B::BlockTensorMap{S}, pB::Index2Tuple,
                                     conjB::Symbol) where {S,N₁,N₂}
    spaces1 = TO.flag2op(conjA).(space.(Ref(A), pA[1]))
    spaces2 = TO.flag2op(conjB).(space.(Ref(B), pB[2]))
    spaces = (spaces1..., spaces2...)
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(spaces), pC[2])))
    return dom → cod
end
function TO.tensorcontract_structure(pC::Index2Tuple{N₁,N₂},
                                     A::BlockTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                                     B::AbstractTensorMap{S}, pB::Index2Tuple,
                                     conjB::Symbol) where {S,N₁,N₂}
    spaces1 = TO.flag2op(conjA).(space.(Ref(A), pA[1]))
    spaces2 = TO.flag2op(conjB).(space.(Ref(B), pB[2]))
    spaces = (spaces1..., spaces2...)
    cod = ProductSumSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
    dom = ProductSumSpace{S,N₂}(dual.(getindex.(Ref(spaces), pC[2])))
    return dom → cod
end
function TO.tensorcontract_structure(pC::Index2Tuple{N₁,N₂},
                                     A::AbstractTensorMap{S}, pA::Index2Tuple,
                                     conjA::Symbol,
                                     B::BlockTensorMap{S}, pB::Index2Tuple,
                                     conjB::Symbol) where {S,N₁,N₂}
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

# PlanarOperations
# ----------------

function TK.BraidingTensor(V1::SumSpace{S}, V2::SumSpace{S}) where {S}
    tdst = BlockTensorMap{S,2,2,TK.BraidingTensor{S,Matrix{ComplexF64}}}(undef, V2 ⊗ V1,
                                                                         V1 ⊗ V2)
    for I in CartesianIndices(tdst)
        if I[1] == I[4] && I[2] == I[3]
            V = getsubspace(space(tdst), I)
            @assert domain(V)[2] == codomain(V)[1] && domain(V)[1] == codomain(V)[2]
            tdst[I] = TK.BraidingTensor(V[2], V[1])
        end
    end
    return tdst
end

function TK.planaradd!(C::BlockTensorMap{S,N₁,N₂},
                       A::BlockTensorMap{S},
                       p::Index2Tuple{N₁,N₂},
                       α::Number,
                       β::Number,
                       backend::Backend...) where {S,N₁,N₂}
    scale!(C, β)
    indCinA = linearize(p)
    for (IA, v) in nonzero_pairs(A)
        IC = CartesianIndex(TupleTools.getindices(IA.I, indCinA))
        C[IC] = TK.planaradd!(C[IC], v, p, α, One())
    end
    return C
end

function TK.planaradd!(C::AbstractTensorMap{S,N₁,N₂}, A::BlockTensorMap{S},
                       p::Index2Tuple{N₁,N₂}, α::Number, β::Number,
                       backend::Backend...) where {S,N₁,N₂}
    C′ = convert(BlockTensorMap, C)
    TK.planaradd!(C′, A, p, α, β, backend...)
    return C
end

function TK.planaradd!(C::BlockTensorMap{S,N₁,N₂},
                       A::AbstractTensorMap{S},
                       p::Index2Tuple{N₁,N₂},
                       α::Number,
                       β::Number,
                       backend::Backend...) where {S,N₁,N₂}
    return TK.planaradd!(C, convert(BlockTensorMap, A), p, α, β, backend...)
end

function TK.planartrace!(C::BlockTensorMap{S,N₁,N₂},
                         A::BlockTensorMap{S},
                         p::Index2Tuple{N₁,N₂},
                         q::Index2Tuple{N₃,N₃},
                         α::Number,
                         β::Number,
                         backend::Backend...) where {S,N₁,N₂,N₃}
    scale!(C, β)

    for (IA, v) in nonzero_pairs(A)
        IAc1 = CartesianIndex(getindices(IA.I, q[1]))
        IAc2 = CartesianIndex(getindices(IA.I, q[2]))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(getindices(IA.I, linearize(p)))
        C[IC] = TK.planartrace!(C[IC], v, p, q, α, One())
    end
    return C
end

function TK.planarcontract!(C::BlockTensorMap{S,N₁,N₂},
                            A::BlockTensorMap{S},
                            pA::Index2Tuple,
                            B::BlockTensorMap{S},
                            pB::Index2Tuple,
                            pAB::Index2Tuple{N₁,N₂},
                            α::Number,
                            β::Number,
                            backend::Backend...) where {S,N₁,N₂}
    scale!(C, β)
    keysA = sort!(collect(nonzero_keys(A));
                  by=IA -> CartesianIndex(getindices(IA.I, pA[2])))
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
                        IC = CartesianIndex(getindices(IABo.I, linearize(pAB)))
                        vA = A[IA]
                        C[IC] = TK.planarcontract!(C[IC], vA, pA, vB, pB, pAB, α, One())
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
                        IC = CartesianIndex(getindices(IABo.I, linearize(pAB)))
                        C[IC] = TK.planarcontract!(C[IC], vA, pA, vB, pB, pAB, α, One())
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

# methods for automatically working with TensorMap - BlockTensorMaps
# ------------------------------------------------------------------------

for (T1, T2) in
    ((:AbstractTensorMap, :BlockTensorMap), (:BlockTensorMap, :AbstractTensorMap),
     (:BlockTensorMap, :BlockTensorMap), (:AbstractTensorMap, :AbstractTensorMap))
    if T1 !== :AbstractTensorMap && T2 !== :AbstractTensorMap
        @eval function TO.tensorcontract!(C::AbstractTensorMap, pC::Index2Tuple, A::$T1,
                                          pA::Index2Tuple, conjA::Symbol, B::$T2,
                                          pB::Index2Tuple, conjB::Symbol, α, β::Number,
                                          backend::TO.Backend...)
            C′ = convert(BlockTensorMap, C)
            tensorcontract!(C′, pC, A, pA, conjA, B, pB, conjB, α, β, backend...)
            return C
        end

        @eval function TK.planarcontract!(C::AbstractTensorMap, A::$T1,
                                          pA::Index2Tuple, B::$T2,
                                          pB::Index2Tuple, pAB::Index2Tuple,
                                          α::Number, β::Number,
                                          backend::Backend...)
            C′ = convert(BlockTensorMap, C)
            TK.planarcontract!(C′, A, pA, B, pB, pAB, α, β, backend...)
            return C
        end

        @eval function TO.checkcontractible(tA::$T1, iA::Int, conjA::Symbol,
                                            tB::$T2, iB::Int, conjB::Symbol,
                                            label)
            sA = TO.tensorstructure(tA, iA, conjA)'
            sB = TO.tensorstructure(tB, iB, conjB)
            sA == sB ||
                throw(SpaceMismatch("incompatible spaces for $label: $sA ≠ $sB"))
            return nothing
        end
    end

    if T1 !== T2
        @eval function TO.tensorcontract_type(TC, pC, A::$T1, pA, conjA, B::$T2, pB, conjB)
            return TO.tensorcontract_type(TC, pC, convert(BlockTensorMap, A), pA, conjA,
                                          convert(BlockTensorMap, B), pB, conjB)
        end
    end

    if !(T1 === :BlockTensorMap && T2 === :BlockTensorMap)
        @eval function TO.tensorcontract!(C::BlockTensorMap, pC::Index2Tuple, A::$T1,
                                          pA::Index2Tuple, conjA::Symbol, B::$T2,
                                          pB::Index2Tuple, conjB::Symbol, α::Number,
                                          β::Number, backend::TO.Backend...)
            return TO.tensorcontract!(C, pC, convert(BlockTensorMap, A), pA, conjA,
                                      convert(BlockTensorMap, B), pB, conjB, α, β,
                                      backend...)
        end

        @eval function TK.planarcontract!(C::BlockTensorMap, A::$T1,
                                          pA::Index2Tuple, B::$T2,
                                          pB::Index2Tuple, pAB::Index2Tuple,
                                          α::Number, β::Number,
                                          backend::Backend...)
            return TK.planarcontract!(C, convert(BlockTensorMap, A), pA,
                                      convert(BlockTensorMap, B), pB, pAB, α, β,
                                      backend...)
        end
    end
end

# TODO: similar for tensoradd!, tensortrace!, planaradd!, planartrace!
