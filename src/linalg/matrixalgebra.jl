# TODO: is it possible to avoid making this contiguous?
function MatrixAlgebra.leftorth!(A::BlockMatrix, alg, atol::Real)
    return MatrixAlgebra.leftorth!(Array(A), alg, atol)
end
function MatrixAlgebra.leftnull!(A::BlockMatrix, alg, atol::Real)
    return MatrixAlgebra.leftnull!(Array(A), alg, atol)
end

function MatrixAlgebra.rightorth!(A::BlockMatrix, alg, atol::Real)
    return MatrixAlgebra.rightorth!(Array(A), alg, atol)
end
function MatrixAlgebra.rightnull!(A::BlockMatrix, alg, atol::Real)
    return MatrixAlgebra.rightnull!(Array(A), alg, atol)
end

function MatrixAlgebra.one!(A::BlockMatrix)
    @inbounds for i in axes(A, 1), j in axes(A, 2)
        A[i, j] = i == j ? one(eltype(A)) : zero(eltype(A))
    end
    return A
end

MatrixAlgebra.svd!(A::BlockMatrix, alg) = MatrixAlgebra.svd!(Array(A), alg)

# piracy
LinearAlgebra.isposdef!(A::BlockMatrix) = LinearAlgebra.isposdef!(Array(A))
