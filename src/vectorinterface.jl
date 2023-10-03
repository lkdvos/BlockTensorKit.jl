VectorInterface.scalartype(T::Type{<:BlockTensorMap}) = scalartype(eltype(T))
VectorInterface.scalartype(x::Union) = Union{scalartype(x.a),scalartype(x.b)} # type piracy!
VectorInterface.scale!(t::BlockTensorMap, α::Number) = scale!.(parent(t), α)
function VectorInterface.scale!(ty::BlockTensorMap, tx::BlockTensorMap, α::Number)
    return scale!.(parent(ty), parent(tx), α)
end
