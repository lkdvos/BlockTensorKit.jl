# TensorOperations
# ----------------

function TO.tensoradd_type(
    TC, A::BlockTensorMap, ::Index2Tuple{N₁,N₂}, ::Bool
) where {N₁,N₂}
    M = TK.similarstoragetype(eltype(A), TC)
    return blocktensormaptype(spacetype(A), N₁, N₂, M)
end
function TO.tensoradd_type(
    TC, A::SparseBlockTensorMap, ::Index2Tuple{N₁,N₂}, ::Bool
) where {N₁,N₂}
    M = TK.similarstoragetype(eltype(A), TC)
    return sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
end
function TO.tensoradd_type(TC, A::AdjointBlockTensorMap, pA::Index2Tuple, conjA::Bool)
    return TO.tensoradd_type(TC, A', adjointtensorindices(A, pA), !conjA)
end

function TO.tensorcontract_type(
    TC,
    A::AbstractBlockTensorMap,
    ::Index2Tuple,
    ::Bool,
    B::AbstractBlockTensorMap,
    ::Index2Tuple,
    ::Bool,
    ::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    spacetype(A) == spacetype(B) ||
        throw(SpaceMismatch("incompatible space types: $(spacetype(A)) ≠ $(spacetype(B))"))
    M = promote_storagetype(TC, eltype(A), eltype(B))

    return if issparse(A) && issparse(B)
        sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
    else
        blocktensormaptype(spacetype(A), N₁, N₂, M)
    end
end
function TO.tensorcontract_type(
    TC,
    A::AbstractTensorMap,
    pA::Index2Tuple,
    conjA::Bool,
    B::AbstractBlockTensorMap,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    spacetype(A) == spacetype(B) ||
        throw(SpaceMismatch("incompatible space types: $(spacetype(A)) ≠ $(spacetype(B))"))
    M = promote_storagetype(TC, typeof(A), eltype(B))
    return if issparse(A) && issparse(B)
        sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
    else
        blocktensormaptype(spacetype(A), N₁, N₂, M)
    end
end
function TO.tensorcontract_type(
    TC,
    A::AbstractBlockTensorMap,
    ::Index2Tuple,
    ::Bool,
    B::AbstractTensorMap,
    ::Index2Tuple,
    ::Bool,
    ::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    spacetype(A) == spacetype(B) ||
        throw(SpaceMismatch("incompatible space types: $(spacetype(A)) ≠ $(spacetype(B))"))
    M = promote_storagetype(TC, eltype(A), typeof(B))
    return if issparse(A) && issparse(B)
        sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
    else
        blocktensormaptype(spacetype(A), N₁, N₂, M)
    end
end

function promote_storagetype(::Type{T}, ::Type{T₁}, ::Type{T₂}) where {T,T₁,T₂}
    M = TK.similarstoragetype(T₁, T)
    @assert M === TK.similarstoragetype(T₂, T) "incompatible storage types"
    # TODO: actually make this work, probably with some promotion rules?
    return M
end
function TK.storagetype(::Type{AbstractTensorMap{E,S,N1,N2}}) where {E,S,N1,N2}
    return Matrix{E}
end

function promote_blocktype(::Type{TT}, ::Type{A₁}, ::Type{A₂}) where {TT,A₁,A₂}
    N = similarblocktype(A₁, TT)
    @assert N === similarblocktype(A₂, TT) "incompatible block types"
    return N
end

function similarblocktype(::Type{A}, ::Type{TT}) where {A,TT}
    return Core.Compiler.return_type(similar, Tuple{A,Type{TT},NTuple{numind(TT),Int}})
end

# By default, make "dense" allocations
function TO.tensoralloc(
    ::Type{BT}, structure::TensorMapSumSpace, istemp::Val, allocator=TO.DefaultAllocator()
) where {BT<:AbstractBlockTensorMap}
    C = BT(undef, structure)
    blockallocator(V) = TO.tensoralloc(eltype(C), V, istemp, allocator)
    map!(blockallocator, parent(C), eachspace(C))
    return C
end

function TO.tensorfree!(t::BlockTensorMap, allocator=TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), parent(t))
    return nothing
end
function TO.tensorfree!(t::SparseBlockTensorMap, allocator=TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), nonzero_values(t))
    return nothing
end

function TK.trace_permute!(
    tdst::BlockTensorMap,
    tsrc::BlockTensorMap,
    (p₁, p₂)::Index2Tuple,
    (q₁, q₂)::Index2Tuple,
    α::Number,
    β::Number,
    backend::AbstractBackend=TO.DefaultBackend(),
)
    @boundscheck begin
        space(tdst) == permute(space(tsrc), (p₁, p₂)) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, q₁[i]) == dual(space(tsrc, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q₁ = $(q₁), q₂ = $(q₂)"))
    end

    szA(i) = size(A, i)
    stA(i) = stride(A, i)
    newstrides = (stA.(linearize((p₁, p₂)))..., (stA.(q₁) .+ stA.(q₂))...)
    newsizes = (size(C)..., szA.(q₁))
    A = StridedView(parent(tsrc), newsizes, newstrides)
    C = StridedView(parent(tdst))

    trace_indices = Iterators.product(ntuple(i -> axes(A, i + ndims(C)), length(q₁))...)
    for I in eachindex(C)
        trace_permute!(C[IC], A[I, first(trace_indices)], (p₁, p₂), (q₁, q₂), α, β, backend)
        for J in tail(trace_indices)
            trace_permute!(C[IC], A[I, J], (p₁, p₂), (q₁, q₂), α, One(), backend)
        end
    end
    return tdst
end

function TO.tensoralloc_contract(TC,
    A::AbstractBlockTensorMap, pA::Index2Tuple, conjA::Bool,
    B::AbstractBlockTensorMap, pB::Index2Tuple, conjB::Bool,
    pAB::Index2Tuple, istemp::Val=Val(false),
    allocator=TO.DefaultAllocator())

    ttype = TO.tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    TT = promote_type(eltype(A), eltype(B))

    if isabstracttype(TT)
        # do not allocate, use undef allocator
        E, S, N1, N2 = scalartype(TT), spacetype(TT), numout(structure), numin(structure)
        if issparse(A) && issparse(B)
            return SparseBlockTensorMap{AbstractTensorMap{E, S, N1, N2}}(undef, codomain(structure), domain(structure))
        else
            return BlockTensorMap{AbstractTensorMap{E, S, N1, N2}}(undef, codomain(structure), domain(structure))
        end
    else
        return tensoralloc(ttype, structure, istemp, allocator)
    end
end

# PlanarOperations
# ----------------

function TK.BraidingTensor(V1::SumSpace{S}, V2::SumSpace{S}) where {S}
    τtype = if BraidingStyle(sectortype(S)) isa SymmetricBraiding
        BraidingTensor{Float64,S}
    else
        BraidingTensor{ComplexF64,S}
    end
    tdst = SparseBlockTensorMap{τtype}(undef, V2 ⊗ V1, V1 ⊗ V2)
    Vs = eachspace(tdst)
    for I in CartesianIndices(tdst)
        if I[1] == I[4] && I[2] == I[3]
            V = Vs[I]
            @assert domain(V)[2] == codomain(V)[1] && domain(V)[1] == codomain(V)[2]
            tdst[I] = TK.BraidingTensor(V[2], V[1])
        end
    end
    return tdst
end

# function TK.planartrace!(C::BlockTensorMap,
#                          A::BlockTensorMap,
#                          p::Index2Tuple,
#                          q::Index2Tuple,
#                          α::Number,
#                          β::Number,
#                          backend::AbstractBackend, allocator)
#     scale!(C, β)
#
#     for (IA, v) in nonzero_pairs(A)
#         IAc1 = CartesianIndex(getindices(IA.I, q[1]))
#         IAc2 = CartesianIndex(getindices(IA.I, q[2]))
#         IAc1 == IAc2 || continue
#
#         IC = CartesianIndex(getindices(IA.I, linearize(p)))
#         C[IC] = TK.planartrace!(C[IC], v, p, q, α, One(), backend, allocator)
#     end
#     return C
# end

# function TK.planarcontract!(C::BlockTensorMap{E,S,N₁,N₂},
#                             A::BlockTensorMap{E,S},
#                             pA::Index2Tuple,
#                             B::BlockTensorMap{E,S},
#                             pB::Index2Tuple,
#                             pAB::Index2Tuple{N₁,N₂},
#                             α::Number,
#                             β::Number,
#                             backend::Backend...) where {E,S,N₁,N₂}
#     scale!(C, β)
#     keysA = sort!(collect(nonzero_keys(A));
#                   by=IA -> CartesianIndex(getindices(IA.I, pA[2])))
#     keysB = sort!(collect(nonzero_keys(B));
#                   by=IB -> CartesianIndex(getindices(IB.I, pB[1])))
#
#     iA = iB = 1
#     @inbounds while iA <= length(keysA) && iB <= length(keysB)
#         IA = keysA[iA]
#         IB = keysB[iB]
#         IAc = CartesianIndex(getindices(IA.I, pA[2]))
#         IBc = CartesianIndex(getindices(IB.I, pB[1]))
#         if IAc == IBc
#             Ic = IAc
#             jA = iA
#             while jA < length(keysA)
#                 if CartesianIndex(getindices(keysA[jA + 1].I, pA[2])) == Ic
#                     jA += 1
#                 else
#                     break
#                 end
#             end
#             jB = iB
#             while jB < length(keysB)
#                 if CartesianIndex(getindices(keysB[jB + 1].I, pB[1])) == Ic
#                     jB += 1
#                 else
#                     break
#                 end
#             end
#             rA = iA:jA
#             rB = iB:jB
#             if length(rA) < length(rB)
#                 for kB in rB
#                     IB = keysB[kB]
#                     IBo = CartesianIndex(getindices(IB.I, pB[2]))
#                     vB = B[IB]
#                     for kA in rA
#                         IA = keysA[kA]
#                         IAo = CartesianIndex(getindices(IA.I, pA[1]))
#                         IABo = CartesianIndex(IAo, IBo)
#                         IC = CartesianIndex(getindices(IABo.I, linearize(pAB)))
#                         vA = A[IA]
#                         C[IC] = TK.planarcontract!(C[IC], vA, pA, vB, pB, pAB, α, One())
#                     end
#                 end
#             else
#                 for kA in rA
#                     IA = keysA[kA]
#                     IAo = CartesianIndex(getindices(IA.I, pA[1]))
#                     vA = A[IA]
#                     for kB in rB
#                         IB = keysB[kB]
#                         IBo = CartesianIndex(getindices(IB.I, pB[2]))
#                         vB = parent(B).data[IB]
#                         IABo = CartesianIndex(IAo, IBo)
#                         IC = CartesianIndex(getindices(IABo.I, linearize(pAB)))
#                         C[IC] = TK.planarcontract!(C[IC], vA, pA, vB, pB, pAB, α, One())
#                     end
#                 end
#             end
#             iA = jA + 1
#             iB = jB + 1
#         elseif IAc < IBc
#             iA += 1
#         else
#             iB += 1
#         end
#     end
#
#     return C
# end
