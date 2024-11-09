"""
    AbstractBlockTensorMap{E,S,N₁,N₂}

Abstract supertype for tensor maps that have additional block structure, i.e. they act on vector spaces
that have a direct sum structure. These behave like `AbstractTensorMap` but have additional methods to
facilitate indexing and manipulation of the block structure.
"""
abstract type AbstractBlockTensorMap{E,S,N₁,N₂} <: AbstractTensorMap{E,S,N₁,N₂} end

include("abstractarray.jl")
include("abstracttensormap.jl")
include("conversion.jl")
include("show.jl")

# Sparsity
# --------
nonzero_pairs(t::AbstractBlockTensorMap) = nonzero_pairs(parent(t))
nonzero_keys(t::AbstractBlockTensorMap) = nonzero_keys(parent(t))
nonzero_values(t::AbstractBlockTensorMap) = nonzero_values(parent(t))
nonzero_length(t::AbstractBlockTensorMap) = nonzero_length(parent(t))

nonzero_values(A::AbstractArray) = values(A)
nonzero_keys(A::AbstractArray) = keys(A)
nonzero_pairs(A::AbstractArray) = pairs(A)
nonzero_length(A::AbstractArray) = length(A)

issparse(t::AbstractTensorMap) = false
issparse(t::TensorKit.AdjointTensorMap) = issparse(parent(t))