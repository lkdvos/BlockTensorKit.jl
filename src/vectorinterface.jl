VectorInterface.scalartype(T::Type{<:BlockTensorMap}) = scalartype(eltype(T))
VectorInterface.scalartype(x::Union) = Union{scalartype(x.a),scalartype(x.b)} # type piracy!
VectorInterface.scale!(t::BlockTensorMap, α::Number) = scale!.(parent(t), α)
function VectorInterface.scale!(ty::BlockTensorMap, tx::BlockTensorMap, α::Number)
    return scale!.(parent(ty), parent(tx), α)
end

function VectorInterface.zerovector(t::BlockTensorMap)
    return zerovector!(similar(t))
end

function VectorInterface.zerovector!(t::BlockTensorMap{S,N1,N2,A}) where {S,N1,N2,A<:SparseArray}
    zerovector!(parent(t))
    return t
end

function VectorInterface.zerovector!(t::BlockTensorMap{S,N1,N2,A}) where {S,N1,N2,A<:AbstractArray}
    for i in eachindex(t)
        if isassigned(t, i)
            zerovector!(t[i])
        else
            V = getsubspace(t, i)
            t[i] = TensorMap(zeros, scalartype(t), V)
        end
    end
    return t
end