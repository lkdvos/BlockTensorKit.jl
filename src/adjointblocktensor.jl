const AdjointBlockTensorMap{T,S,N₁,N₂,TT<:AbstractBlockTensorMap} = AdjointTensorMap{
    T,S,N₁,N₂,TT
}

function permute_adjointindices(t::AbstractTensorMap, I::CartesianIndex)
    return CartesianIndex(
        TupleTools.getindices(I.I, adjointtensorindices(t, ntuple(identity, length(I.I))))
    )
end
function nonzero_pairs(t::AdjointBlockTensorMap)
    return (permute_adjointindices(t, I) => v' for (I, v) in nonzero_pairs(t'))
end
function nonzero_keys(t::AdjointBlockTensorMap)
    return (permute_adjointindices(t, I) for I in nonzero_keys(t'))
end
function nonzero_values(t::AdjointBlockTensorMap)
    return (v' for v in nonzero_values(t'))
end
function nonzero_length(t::AdjointBlockTensorMap)
    return nonzero_length(t')
end
