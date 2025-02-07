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
            tdst[I′], v, (p₁, p₂), fusiontreetransform, α, One(), backend...
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
            tdst[I′], v, (p₁, p₂), fusiontreetransform, α, One(), backend...
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

# we need to capture the other functions earlier to enjoy the fast transformers...
for f! in (:add_permute!, :add_transpose!)
    @eval function TK.$f!(
        tdst::BlockTensorMap,
        tsrc::BlockTensorMap,
        (p₁, p₂)::Index2Tuple{N₁,N₂},
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
            dstdata[I] = TK.$f!(dstdata[I], srcdata[I], (p₁, p₂), α, β, backend...)
        end
        return tdst
    end
    @eval function TK.$f!(
        tdst::AbstractBlockTensorMap,
        tsrc::AbstractBlockTensorMap,
        (p₁, p₂)::Index2Tuple{N₁,N₂},
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
            tdst[I′] = TK.$f!(tdst[I′], v, (p₁, p₂), α, One(), backend...)
        end
        return tdst
    end
    @eval function TK.$f!(
        tdst::AbstractBlockTensorMap,
        tsrc::AdjointTensorMap{T,S,N₁,N₂,TT},
        (p₁, p₂)::Index2Tuple,
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
            tdst[I′] = TK.$f!(tdst[I′], v, (p₁, p₂), α, One(), backend...)
        end
        return tdst
    end
    @eval function TK.$f!(
        tdst::TensorMap,
        tsrc::BlockTensorMap,
        (p₁, p₂)::Index2Tuple,
        α::Number,
        β::Number,
        backend::AbstractBackend...,
    )
        @assert length(tsrc) == 1 "source tensor must be a single tensor"
        return TK.$f!(tdst, only(tsrc), (p₁, p₂), α, β, backend...)
    end
end

function TK.add_braid!(
    tdst::BlockTensorMap,
    tsrc::BlockTensorMap,
    (p₁, p₂)::Index2Tuple{N₁,N₂},
    levels::IndexTuple,
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
        dstdata[I] = TK.add_braid!(
            dstdata[I], srcdata[I], (p₁, p₂), levels, α, β, backend...
        )
    end
    return tdst
end
function TK.add_braid!(
    tdst::AbstractBlockTensorMap,
    tsrc::AbstractBlockTensorMap,
    (p₁, p₂)::Index2Tuple{N₁,N₂},
    levels::IndexTuple,
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
        tdst[I′] = TK.add_braid!(tdst[I′], v, (p₁, p₂), levels, α, One(), backend...)
    end
    return tdst
end
function TK.add_braid!(
    tdst::AbstractBlockTensorMap,
    tsrc::AdjointTensorMap{T,S,N₁,N₂,TT},
    (p₁, p₂)::Index2Tuple,
    levels::IndexTuple,
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
        tdst[I′] = TK.add_braid!(tdst[I′], v, (p₁, p₂), levels, α, One(), backend...)
    end
    return tdst
end
function TK.add_braid!(
    tdst::TensorMap,
    tsrc::BlockTensorMap,
    (p₁, p₂)::Index2Tuple,
    levels::IndexTuple,
    α::Number,
    β::Number,
    backend::AbstractBackend...,
)
    @assert length(tsrc) == 1 "source tensor must be a single tensor"
    return TK.add_braid!(tdst, only(tsrc), (p₁, p₂), levels, α, β, backend...)
end

Base.@constprop :aggressive function TK.removeunit(
    t::SparseBlockTensorMap, i::Int; copy::Bool=false
)
    W = removeunit(space(t), i)
    tdst = similar(t, W)
    for (I, v) in nonzero_pairs(t)
        I′ = CartesianIndex(TupleTools.deleteat(I.I, i))
        tdst[I′] = removeunit(v, i; copy)
    end
    return tdst
end
