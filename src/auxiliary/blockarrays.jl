copy_dense(A) = copy_dense!(similar(first(A.blocks), size(A)), A)
function copy_dense!(Adense, A)
    for bj in blockaxes(A, 2)
        js = axes(A, 2)[bj]
        for bi in blockaxes(A, 1)
            a = view(A, bi, bj)
            is = axes(A, 1)[bi]
            Adense[is, js] = @view A[block_index...]
        end
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
