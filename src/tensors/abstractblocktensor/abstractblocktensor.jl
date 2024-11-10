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
include("sparsity.jl")