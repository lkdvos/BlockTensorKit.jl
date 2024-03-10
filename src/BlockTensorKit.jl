module BlockTensorKit

export SumSpace
export BlockTensorMap, SparseBlockTensorMap
export undef_blocks
export getsubspace

using TensorKit
using TensorKit: OneOrNoneIterator
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
include("blocktensor.jl")

# various interfaces
include("linalg.jl")
include("tensorkit.jl")
include("vectorinterface.jl")
include("tensoroperations.jl")
export BlockTensorMap
export nonzero_keys, nonzero_values, nonzero_pairs, nonzero_length

end
