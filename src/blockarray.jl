const BlockSparseArray{T, N, A} = BlockArray{T, N, SparseArray{A,N}} where {T, N, A<:AbstractArray{T,N}}

Base.@propagate_inbounds function BlockArrays.viewblock(block_arr::BlockSparseArray{T,N,A},
                                                        block::Block{N}) where {T,N,A}
    blks = block.n
    @boundscheck blockcheckbounds(block_arr, blks...)
    return get(block_arr.blocks.data, CartesianIndex(blks...)) do 
        return fill!(similar(A, map(getindex, blocksizes(block_arr), blks)), zero(T))
    end
end

Base.:(\)(b1::BlockSparseArray{T,N,A}, b2::BlockSparseArray{T,N,A}) where {T,N,A} = PseudoBlockArray(b1) \ PseudoBlockArray(b2)

function Base.:(/)(b1::BlockSparseArray{T,N,A}, b2::BlockSparseArray{T,N,A}) where {T,N,A}
    return PseudoBlockArray(b1) / PseudoBlockArray(b2)
end

# Base.adjoint(A::PseudoBlockArray) = PseudoBlockArray(adjoint(A.blocks), A.axes)

# function LinearAlgebra.ldiv!(A::Adjoint{<:Any,<:Union{LinearAlgebra.QR,LinearAlgebra.QRCompactWY,LinearAlgebra.QRPivoted}},
#                              B::PseudoBlockMatrix)
#     return ldiv!(A, B.blocks)
# end
