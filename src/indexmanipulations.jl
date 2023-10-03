function TK.add_permute!(tdst::BlockTensorMap{S,N₁,N₂}, tsrc::BlockTensorMap{S},
                         (p1, p2)::Index2Tuple{N₁,N₂}, α::Number, β::Number,
                         backend::TO.Backend...) where {S,N₁,N₂}
    for I in eachindex(tsrc)
        I′ = CartesianIndex(getindex.(Ref(I), (p1..., p2...)))
        tdst[I′] = TK.add_permute!(tdst[I′], tsrc[I], (p1, p2), α, β, backend...)
    end
    return tdst
end
