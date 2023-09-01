module BlockTensorKit

using TensorKit
using SparseArrayKit
using VectorInterface
using TensorOperations
using LinearAlgebra
using Strided
using TupleTools: getindices, isperm
using BlockArrays
using TupleTools

import VectorInterface

# Spaces
include("sumspace.jl")
export SumSpace

# TensorMaps
include("blockarray.jl")
include("blocktensor.jl")
include("sparseblocktensor.jl")
export BlockTensorMap

# implementations
include("linalg.jl")
include("vectorinterface.jl")
include("tensorinterface.jl")

end
