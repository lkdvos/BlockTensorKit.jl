@noinline function _check_spacetype(
        ::Type{S₁}, ::Type{S₂}
    ) where {S₁ <: ElementarySpace, S₂ <: ElementarySpace}
    S₁ === S₂ ||
        S₁ === SumSpace{S₂} ||
        SumSpace{S₁} === S₂ ||
        throw(SpaceMismatch(lazy"incompatible spacetypes: $S₁ and $S₂"))
    return nothing
end

# TensorOperations
# ----------------
function TO.tensoradd_type(
        TC, A::AbstractBlockTensorMap, ::Index2Tuple{N₁, N₂}, ::Bool
    ) where {N₁, N₂}
    TA = eltype(A)
    I = sectortype(A)
    Tnew = sectorscalartype(I) <: Real ? TC : complex(TC)
    if TA isa Union
        M = Union{TK.similarstoragetype(TA.a, Tnew), TK.similarstoragetype(TA.b, Tnew)}
    else
        M = TK.similarstoragetype(TA, Tnew)
    end
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
        A::AbstractBlockTensorMap, ::Index2Tuple, ::Bool,
        B::AbstractBlockTensorMap, ::Index2Tuple, ::Bool,
        ::Index2Tuple{N₁, N₂},
    ) where {N₁, N₂}
    _check_spacetype(spacetype(A), spacetype(B))

    I = sectortype(A)
    Tnew = sectorscalartype(I) <: Real ? TC : complex(TC)
    M = promote_storagetype(Tnew, eltype(A), eltype(B))

    return if issparse(A) && issparse(B)
        sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
    else
        blocktensormaptype(spacetype(A), N₁, N₂, M)
    end
end
function TO.tensorcontract_type(
        TC,
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractBlockTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple{N₁, N₂},
    ) where {N₁, N₂}
    _check_spacetype(spacetype(A), spacetype(B))

    I = sectortype(A)
    Tnew = sectorscalartype(I) <: Real ? TC : complex(TC)
    M = promote_storagetype(Tnew, typeof(A), eltype(B))

    return if issparse(A) && issparse(B)
        sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
    else
        blocktensormaptype(spacetype(A), N₁, N₂, M)
    end
end
function TO.tensorcontract_type(
        TC,
        A::AbstractBlockTensorMap, ::Index2Tuple, ::Bool,
        B::AbstractTensorMap, ::Index2Tuple, ::Bool,
        ::Index2Tuple{N₁, N₂},
    ) where {N₁, N₂}
    _check_spacetype(spacetype(A), spacetype(B))

    I = sectortype(A)
    Tnew = sectorscalartype(I) <: Real ? TC : complex(TC)
    M = promote_storagetype(Tnew, eltype(A), typeof(B))

    return if issparse(A) && issparse(B)
        sparseblocktensormaptype(spacetype(A), N₁, N₂, M)
    else
        blocktensormaptype(spacetype(A), N₁, N₂, M)
    end
end

function TO.tensoralloc_contract(
        TC,
        A::AbstractBlockTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractBlockTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        istemp::Val = Val(false),
        allocator = TO.DefaultAllocator(),
    )
    ttype = TO.tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    TT = eltype(ttype)

    if isabstracttype(TT)
        # do not allocate, use undef allocator
        E, S, N1, N2 = scalartype(TT), spacetype(TT), numout(structure), numin(structure)
        if issparse(A) && issparse(B)
            return SparseBlockTensorMap{AbstractTensorMap{E, S, N1, N2}}(
                undef, codomain(structure), domain(structure)
            )
        else
            return BlockTensorMap{AbstractTensorMap{E, S, N1, N2}}(
                undef, codomain(structure), domain(structure)
            )
        end
    else
        return tensoralloc(ttype, structure, istemp, allocator)
    end
end

function promote_storagetype(::Type{T}, ::Type{T₁}, ::Type{T₂}) where {T, T₁, T₂}
    if T₁ isa Union
        M₁ = Union{TK.similarstoragetype(T₁.a, T), TK.similarstoragetype(T₁.b, T)}
    else
        M₁ = TK.similarstoragetype(T₁, T)
    end
    if T₂ isa Union
        M₂ = Union{TK.similarstoragetype(T₂.a, T), TK.similarstoragetype(T₂.b, T)}
    else
        M₂ = TK.similarstoragetype(T₂, T)
    end
    return Union{M₁, M₂}
end

# EVIL HACK!!!
TK.storagetype(::Type{AbstractTensorMap{TT, S, N₁, N₂}}) where {TT, S, N₁, N₂} = Vector{TT}

function promote_blocktype(::Type{TT}, ::Type{A₁}, ::Type{A₂}) where {TT, A₁, A₂}
    N = similarblocktype(A₁, TT)
    @assert N === similarblocktype(A₂, TT) "incompatible block types"
    return N
end

function similarblocktype(::Type{A}, ::Type{TT}) where {A, TT}
    return Core.Compiler.return_type(similar, Tuple{A, Type{TT}, NTuple{numind(TT), Int}})
end

# By default, make "dense" allocations
function TO.tensoralloc(
        ::Type{BT}, structure::TensorMapSumSpace, istemp::Val, allocator = TO.DefaultAllocator()
    ) where {BT <: AbstractBlockTensorMap}
    C = BT(undef_blocks, structure)
    blockallocator(V) = TO.tensoralloc(eltype(C), V, istemp, allocator)
    map!(blockallocator, parent(C), eachspace(C))
    return C
end

# unfortunate overlaod until TK fix
function TK.blas_contract!(
        C::AbstractBlockTensorMap,
        A::AbstractBlockTensorMap, pA::Index2Tuple,
        B::AbstractBlockTensorMap, pB::Index2Tuple,
        pAB::Index2Tuple, α, β,
        backend, allocator
    )
    bstyle = BraidingStyle(sectortype(C))
    bstyle isa SymmetricBraiding ||
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    TC = scalartype(C)

    # check which tensors have to be permuted/copied
    copyA = !(TO.isblascontractable(A, pA) && eltype(A) === TC)
    copyB = !(TO.isblascontractable(B, pB) && eltype(B) === TC)

    if bstyle isa Fermionic && any(isdual ∘ Base.Fix1(space, B), pB[1])
        # twist smallest object if neither or both already have to be permuted
        # otherwise twist the one that already is copied
        if !(copyA ⊻ copyB)
            twistA = dim(A) < dim(B)
        else
            twistA = copyA
        end
        twistB = !twistA
        copyA |= twistA
        copyB |= twistB
    else
        twistA = false
        twistB = false
    end

    # Bring A in the correct form for BLAS contraction
    if copyA
        Anew = TO.tensoralloc_add(TC, A, pA, false, Val(true), allocator)
        Anew = TO.tensoradd!(Anew, A, pA, false, One(), Zero(), backend, allocator)
        twistA && twist!(Anew, filter(!isdual ∘ Base.Fix1(space, Anew), domainind(Anew)))
    else
        Anew = permute(A, pA)
    end
    pAnew = (codomainind(Anew), domainind(Anew))

    # Bring B in the correct form for BLAS contraction
    if copyB
        Bnew = TO.tensoralloc_add(TC, B, pB, false, Val(true), allocator)
        Bnew = TO.tensoradd!(Bnew, B, pB, false, One(), Zero(), backend, allocator)
        twistB && twist!(Bnew, filter(isdual ∘ Base.Fix1(space, Bnew), codomainind(Bnew)))
    else
        Bnew = permute(B, pB)
    end
    pBnew = (codomainind(Bnew), domainind(Bnew))

    # Bring C in the correct form for BLAS contraction
    ipAB = TO.oindABinC(pAB, pAnew, pBnew)
    copyC = !TO.isblasdestination(C, ipAB)

    if copyC
        Cnew = TO.tensoralloc_add(TC, C, ipAB, false, Val(true), allocator)
        mul!(Cnew, Anew, Bnew)
        TO.tensoradd!(C, Cnew, pAB, false, α, β, backend, allocator)
        TO.tensorfree!(Cnew, allocator)
    else
        Cnew = permute(C, ipAB)
        mul!(Cnew, Anew, Bnew, α, β)
    end

    copyA && TO.tensorfree!(Anew, allocator)
    copyB && TO.tensorfree!(Bnew, allocator)

    return C
end


# tensorfree!
# -----------
function TO.tensorfree!(t::BlockTensorMap, allocator = TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), parent(t))
    return nothing
end
function TO.tensorfree!(t::SparseBlockTensorMap, allocator = TO.DefaultAllocator())
    foreach(Base.Fix2(TO.tensorfree!, allocator), nonzero_values(t))
    return nothing
end

function TK.trace_permute!(
        tdst::AbstractBlockTensorMap,
        tsrc::AbstractBlockTensorMap,
        (p₁, p₂)::Index2Tuple,
        (q₁, q₂)::Index2Tuple,
        α::Number, β::Number,
        backend::AbstractBackend = TO.DefaultBackend(),
    )
    # some input checks
    _check_spacetype(spacetype(tdst), spacetype(tsrc))
    if !(BraidingStyle(sectortype(tdst)) isa SymmetricBraiding)
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

function TK.BraidingTensor(
        V1::SumSpace{S}, V2::SumSpace{S}, adjoint::Bool = false
    ) where {S}
    T = BraidingStyle(sectortype(S)) isa SymmetricBraiding ? Float64 : ComplexF64
    return TK.BraidingTensor{T, S}(V1, V2, adjoint)
end

function TK.BraidingTensor{T, S}(
        V1::SumSpace{S}, V2::SumSpace{S}, adjoint::Bool = false
    ) where {T, S}
    τtype = BraidingTensor{T, S}
    tdst = SparseBlockTensorMap{τtype}(undef, V2 ⊗ V1, V1 ⊗ V2)
    Vs = eachspace(tdst)
    @inbounds for I in CartesianIndices(tdst)
        if I[1] == I[4] && I[2] == I[3]
            V = Vs[I]
            tdst[I] = TK.BraidingTensor{T, S}(V[2], V[1], adjoint)
        end
    end
    return tdst
end
