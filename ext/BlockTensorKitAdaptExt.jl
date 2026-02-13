module BlockTensorKitAdaptExt

using TensorKit
using BlockTensorKit
using Adapt

function Adapt.adapt_structure(to, x::BlockTensorMap)
    data′ = map(adapt(to), x.data)
    return BlockTensorMap(data′, space(x))
end

function Adapt.adapt_structure(to, x::SparseBlockTensorMap)
    ad = adapt(to)
    TT = Base.promote_op(ad, eltype(x))
    data′ = Dict{CartesianIndex{ndims(x)}, TT}(I => adapt(to, v) for (I, v) in x.data)
    return SparseBlockTensorMap(data′, space(x))
end

end
