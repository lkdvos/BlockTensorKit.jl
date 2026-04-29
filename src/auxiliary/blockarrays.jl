similar_dense(::Type{Vector{T}}, sz::NTuple{N, Ti}) where {T, N, Ti} = Array{T, N}(undef, sz)
similar_dense(::Type{TA}, sz) where {T, N, P, TA <: Base.SubArray{T, N, P}} = similar_dense(P, sz)
similar_dense(::Type{TA}, sz) where {T, N, P, TA <: Base.ReshapedArray{T, N, P}} = similar_dense(P, sz)
similar_dense(::Type{TA}, sz) where {T, N, TA <: AbstractArray{T, N}} = TA(undef, sz)
function similar_dense(A::BlockMatrix{T, R}) where {T, R}
    Adense = similar_dense(eltype(R), size(A))
    return Adense
end

function copy_dense!(Adense, A)
    for block_index in Iterators.product(blockaxes(A)...)
        a = view(A, block_index...)
        indices = getindex.(axes(A), block_index)
        Adense[indices...] .= a
    end
    return Adense
end

const BlockBlasMat{T <: MAK.BlasFloat} = BlockMatrix{T}

function MAK.zero!(A::BlockBlasMat)
    for bj in blockaxes(A, 2), bi in blockaxes(A, 1)
        a = view(A, bi, bj)
        MAK.zero!(a)
    end
    return A
end

function MAK.one!(A::BlockBlasMat)
    for bj in blockaxes(A, 2), bi in blockaxes(A, 1)
        a = view(A, bi, bj)
        bi == bj ? MAK.one!(a) : MAK.zero!(a)
    end
    return A
end
