module BlockTensorKit

export SumSpace, ProductSumSpace
export eachspace, SumSpaceIndices, sumspacetype

export AbstractBlockTensorMap, BlockTensorMap, SparseBlockTensorMap
export blocktensormaptype, sparseblocktensormaptype

export SparseTensorArray

export sprand, spzeros
export nonzero_keys, nonzero_values, nonzero_pairs, nonzero_length
export sparse
export dropzeros!, droptol!
export undef_blocks

using TensorKit
using TensorKit:
    OneOrNoneIterator,
    HomSpace,
    MatrixAlgebra,
    SectorDict,
    AdjointTensorMap,
    adjointtensorindices,
    compose
using VectorInterface
using TensorOperations
using TensorOperations:
    dimcheck_tensoradd,
    dimcheck_tensorcontract,
    dimcheck_tensortrace,
    argcheck_tensoradd,
    argcheck_tensorcontract,
    argcheck_tensortrace,
    AbstractBackend
using LinearAlgebra
using Strided
using BlockArrays
using BlockArrays: UndefBlocksInitializer
using TupleTools
using Base: @propagate_inbounds
using Random

using Compat

import VectorInterface as VI
import TensorKit as TK
import TensorOperations as TO
import TupleTools as TT

# Spaces
include("vectorspaces/sumspace.jl")
include("vectorspaces/sumspaceindices.jl")

# Tensors
include("tensors/abstractblocktensor/abstractblocktensor.jl")
include("tensors/blocktensor.jl")
include("tensors/sparseblocktensor.jl")
include("tensors/adjointblocktensor.jl")

include("tensors/indexmanipulations.jl")
include("tensors/vectorinterface.jl")
include("tensors/tensoroperations.jl")

include("linalg/linalg.jl")
include("linalg/matrixalgebra.jl")
include("linalg/factorizations.jl")

include("auxiliary/sparsetensorarray.jl")

end
