module BlockTensorKit

export SumSpace
export BlockTensorMap, SparseBlockTensorMap
export undef_blocks
export getsubspace

using TensorKit
using SparseArrayKit
using VectorInterface
using TensorOperations
using TensorOperations: dimcheck_tensoradd, dimcheck_tensorcontract, dimcheck_tensortrace,
                        argcheck_tensoradd, argcheck_tensorcontract, argcheck_tensortrace,
                        Backend
using LinearAlgebra
using Strided
using TupleTools: getindices, isperm
using BlockArrays
using TupleTools
using Base: @propagate_inbounds

import VectorInterface as VI
import TensorKit as TK
import TensorOperations as TO

# Spaces
include("sumspace.jl")
export SumSpace, ProductSumSpace

# TensorMaps
# include("blocktensor.jl")
include("blocktensor.jl")
export BlockTensorMap
# export blocktype

# implementations
# include("indexmanipulations.jl")
# include("linalg.jl")
# include("vectorinterface.jl")
# include("tensoroperations.jl")

# function adjointtensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
#     I = sectortype(S)
#     if T <: DenseMatrix
#         M = T
#     elseif T <: Number
#         M = Matrix{T}
#     else
#         throw(ArgumentError("the final argument of `tensormaptype` should either be the scalar or the storage type, i.e. a subtype of `Number` or of `DenseMatrix`"))
#     end
#     if I === Trivial
#         return AdjointTensorMap{S,N₁,N₂,I,M,Nothing,Nothing}
#     else
#         F₁ = fusiontreetype(I, N₁)
#         F₂ = fusiontreetype(I, N₂)
#         return AdjointTensorMap{S,N₁,N₂,I,SectorDict{I,M},F₁,F₂}
#     end
# end
# adjointtensormaptype(S, N₁, N₂=0) = adjointtensormaptype(S, N₁, N₂, Float64)

end
