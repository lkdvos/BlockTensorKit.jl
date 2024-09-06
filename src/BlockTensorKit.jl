module BlockTensorKit

export SumSpace, ProductSumSpace
export eachspace, SumSpaceIndices, sumspacetype
export BlockTensorMap, SparseBlockTensorMap
export SparseTensorArray
export nonzero_keys, nonzero_values, nonzero_pairs, nonzero_length
export undef_blocks

using TensorKit
using TensorKit: OneOrNoneIterator, HomSpace, MatrixAlgebra, SectorDict
using VectorInterface
using TensorOperations
using TensorOperations: dimcheck_tensoradd, dimcheck_tensorcontract, dimcheck_tensortrace,
                        argcheck_tensoradd, argcheck_tensorcontract, argcheck_tensortrace,
                        AbstractBackend
using LinearAlgebra
using Strided
using TupleTools: getindices, isperm
using BlockArrays
using TupleTools
using Base: @propagate_inbounds
using Random

import VectorInterface as VI
import TensorKit as TK
import TensorOperations as TO

# Spaces
include("sumspace.jl")
include("sumspaceindices.jl")

# Arrays
include("sparsetensorarray.jl")

# TensorMaps
include("blocktensor.jl")

# various interfaces
include("matrixalgebra.jl")
include("linalg.jl")
include("factorizations.jl")
include("tensorkit.jl")
include("vectorinterface.jl")
include("tensoroperations.jl")
include("indexmanipulations.jl")

end
