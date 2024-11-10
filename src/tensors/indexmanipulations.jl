function TK.add_transform!(
    tdst::BlockTensorMap,
    tsrc::BlockTensorMap,
    (p₁, p₂)::Index2Tuple{N₁,N₂},
    fusiontreetransform,
    α::Number,
    β::Number,
    backend::AbstractBackend...,
) where {N₁,N₂}
    @boundscheck begin
        permute(space(tsrc), (p₁, p₂)) == space(tdst) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
    end
    dstdata = parent(tdst)
    srcdata = permutedims(StridedView(parent(tsrc)), (p₁..., p₂...))

    @inbounds for I in eachindex(dstdata, srcdata)
        dstdata[I] = TK.add_transform!(
            dstdata[I], srcdata[I], (p₁, p₂), fusiontreetransform, α, β, backend...
        )
    end
    return tdst
end
function TK.add_transform!(
    tdst::AbstractBlockTensorMap,
    tsrc::AbstractBlockTensorMap,
    (p₁, p₂)::Index2Tuple{N₁,N₂},
    fusiontreetransform,
    α::Number,
    β::Number,
    backend::AbstractBackend...,
) where {N₁,N₂}
    @boundscheck begin
        permute(space(tsrc), (p₁, p₂)) == space(tdst) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
    end
    scale!(tdst, β)
    p = (p₁..., p₂...)
    for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p))
        tdst[I′] = TK.add_transform!(
            tdst[I′], v, (p₁, p₂), fusiontreetransform, α, one(scalartype(tdst)), backend...
        )
    end
    return tdst
end
function TK.add_transform!(
    tdst::AbstractBlockTensorMap,
    tsrc::AdjointTensorMap{T,S,N₁,N₂,TT},
    (p₁, p₂)::Index2Tuple,
    fusiontreetransform,
    α::Number,
    β::Number,
    backend::AbstractBackend...,
) where {T,S,N₁,N₂,TT<:AbstractBlockTensorMap}
    @boundscheck begin
        permute(space(tsrc), (p₁, p₂)) == space(tdst) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
    end
    scale!(tdst, β)
    p = (p₁..., p₂...)
    for (I, v) in nonzero_pairs(tsrc)
        I′ = CartesianIndex(TT.getindices(I.I, p))
        tdst[I′] = TK.add_transform!(
            tdst[I′], v, (p₁, p₂), fusiontreetransform, α, one(scalartype(tdst)), backend...
        )
    end
    return tdst
end
function TK.add_transform!(
    tdst::TensorMap,
    tsrc::BlockTensorMap,
    (p₁, p₂)::Index2Tuple,
    fusiontreetransform,
    α::Number,
    β::Number,
    backend::AbstractBackend...,
)
    @assert length(tsrc) == 1 "source tensor must be a single tensor"
    return TK.add_transform!(
        tdst, only(tsrc), (p₁, p₂), fusiontreetransform, α, β, backend...
    )
end
TK.has_shared_permute(::AbstractBlockTensorMap, args...) = false
