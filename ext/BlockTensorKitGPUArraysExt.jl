module BlockTensorKitGPUArraysExt

using BlockTensorKit, BlockArrays, GPUArrays, Strided
using Strided: StridedViews
using GPUArrays: KernelAbstractions

function KernelAbstractions.get_backend(BA::BlockArrays.BlockArray{T, N, A}) where {T, N, A <: AbstractArray{<:StridedView{T, N, <:AnyGPUArray}}}
    return KernelAbstractions.get_backend(first(BA.blocks))
end

end
