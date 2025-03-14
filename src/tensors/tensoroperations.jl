# TensorOperations
# ----------------
function TO.tensoradd_type(
    TC, A::AbstractBlockTensorMap, ::Index2Tuple{N₁,N₂}, ::Bool
) where {N₁,N₂}
    M = TK.similarstoragetype(eltype(A), TC)
    return if issparse(A)
        sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
    else
        blocktensormaptype(spacetype(A), N₁, N₂, M)
    end
end
function TO.tensoradd_type(TC, A::AdjointBlockTensorMap, pA::Index2Tuple, conjA::Bool)
    return TO.tensoradd_type(TC, A', adjointtensorindices(A, pA), !conjA)
end

# tensoralloc_contract
# --------------------
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

function TO.tensoralloc_contract(
    TC,
    A::AbstractBlockTensorMap,
    pA::Index2Tuple,
    conjA::Bool,
    B::AbstractBlockTensorMap,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple,
    istemp::Val=Val(false),
    allocator=TO.DefaultAllocator(),
)
    ttype = TO.tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    TT = eltype(ttype)

    if isabstracttype(TT)
        # do not allocate, use undef allocator
        E, S, N1, N2 = scalartype(TT), spacetype(TT), numout(structure), numin(structure)
        if issparse(A) && issparse(B)
            return SparseBlockTensorMap{AbstractTensorMap{E,S,N1,N2}}(
                undef, codomain(structure), domain(structure)
            )
        else
            return BlockTensorMap{AbstractTensorMap{E,S,N1,N2}}(
                undef, codomain(structure), domain(structure)
            )
        end
    else
        return tensoralloc(ttype, structure, istemp, allocator)
    end
end

function promote_storagetype(::Type{T}, ::Type{T₁}, ::Type{T₂}) where {T,T₁,T₂}
    M = TK.similarstoragetype(T₁, T)
    @assert M === TK.similarstoragetype(T₂, T) "incompatible storage types"
    # TODO: actually make this work, probably with some promotion rules?
    return M
end
# evil hack???
function TK.storagetype(::Type{AbstractTensorMap{E,S,N1,N2}}) where {E,S,N1,N2}
    return Vector{E}
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
    C = BT(undef_blocks, structure)
    blockallocator(V) = TO.tensoralloc(eltype(C), V, istemp, allocator)
    map!(blockallocator, parent(C), eachspace(C))
    return C
end

# tensorfree!
# -----------
function TO.tensorfree!(t::BlockTensorMap, allocator=TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), parent(t))
    return nothing
end
function TO.tensorfree!(t::SparseBlockTensorMap, allocator=TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), nonzero_values(t))
    return nothing
end

function TK.trace_permute!(
    tdst::AbstractBlockTensorMap,
    tsrc::AbstractBlockTensorMap,
    (p₁, p₂)::Index2Tuple,
    (q₁, q₂)::Index2Tuple,
    α::Number,
    β::Number,
    backend::AbstractBackend=TO.DefaultBackend(),
)
    # some input checks
    (S = spacetype(tdst)) == spacetype(tsrc) ||
        throw(SpaceMismatch("incompatible spacetypes"))
    if !(BraidingStyle(sectortype(S)) isa SymmetricBraiding)
        throw(
            SectorMismatch(
                "only tensors with symmetric braiding rules can be contracted; try `@planar` instead",
            ),
        )
    end
    (N₃ = length(q₁)) == length(q₂) ||
        throw(IndexError("number of trace indices does not match"))

    @boundscheck begin
        space(tdst) == TK.select(space(tsrc), (p₁, p₂)) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, q₁[i]) == dual(space(tsrc, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q₁ = $(q₁), q₂ = $(q₂)"))
    end

    scale!(tdst, β)
    @inbounds for (Isrc, vsrc) in nonzero_pairs(tsrc)
        TT.getindices(Isrc.I, q₁) == TT.getindices(Isrc.I, q₂) || continue
        Idst = CartesianIndex(TT.getindices(Isrc.I, (p₁..., p₂...)))
        tdst[Idst] = TensorKit.trace_permute!(
            tdst[Idst], vsrc, (p₁, p₂), (q₁, q₂), α, One(), backend
        )
    end
    return tdst
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
    @inbounds for I in CartesianIndices(tdst)
        if I[1] == I[4] && I[2] == I[3]
            V = Vs[I]
            @assert domain(V)[2] == codomain(V)[1] && domain(V)[1] == codomain(V)[2]
            tdst[I] = TK.BraidingTensor(V[2], V[1])
        end
    end
    return tdst
end
