## `AbstractBlockTensorMap`

The main type of this package is `BlockTensorMap`, which is a generalization of `AbstractTensorMap` to the case where the tensor is a concatenation of other tensors.
The design philosophy is to have the same interface as `AbstractTensorMap`, with the additional ability to query the individual tensors that make up the block tensor, as `AbstractArray{AbstractTensorMap}`.

### Type hierarchy

The type hierarchy of the `AbstractBlockTensorMap` consists of one abstract and two concrete subtypes of `AbstractBlockTensorMap`:

```julia
BlockTensorMap <: AbstractBlockTensorMap <: AbstractTensorMap
SparseBlockTensorMap <: AbstractBlockTensorMap <: AbstractTensorMap
```

In particular, these structures hold the structural information as a `HomSpace` of `SumSpace`s, as defined in [`SumSpaces`](@ref), as well as the individual tensors that make up the block tensor.
For `BlockTensorMap`, the list of tensors is dense, thus they are stored in an `Array{AbstractTensorMap,N}`, where `N` is the total number of indices of a tensor.
For `SparseBlockTensorMap`, this is not the case, and the list of tensors is stored in a `Dict{CartesianIndex{N},AbstractTensorMap}`.

The elementary constructors for these types are:

```julia
BlockTensorMap{TT}(undef, space::TensorMapSumSpace)
SparseBlockTensorMap{TT}(undef, space::TensorMapSumSpace)
```

where `TT<:AbstractTensorMap` is the type of the individual tensors, and `space` is the `TensorMapSumSpace` that defines the structure of the block tensor.

Similarly, they can be initialized from a list of tensors:

```julia
BlockTensorMap{TT}(tensors::AbstractArray{AbstractTensorMap,N}, space::TensorMapSumSpace)
SparseBlockTensorMap{TT}(tensors::Dict{CartesianIndex{N},AbstractTensorMap}, space::TensorMapSumSpace)
```

!!!note In analogy to `TensorKit`, most of the functionality that requires a `space` object can equally well be called in terms of `codomain(space), domain(space)`, if that is more convenient.

### Indexing

For indexing operators, `AbstractBlockTensorMap` behaves like an `AbstractArray{AbstractTensorMap}`, and the individual tensors can be accessed via the `getindex` and `setindex!` functions.
In particular, the `getindex` function returns a `TT` object, and the `setindex!` function expects a `TT` object.

```julia

```

Slicing operations are also supported, and the `AbstractBlockTensorMap` can be sliced in the same way as an `AbstractArray{AbstractTensorMap}`.
There is however one elementary difference: as the slices still contain tensors with the same amount of legs, there can be no reduction in the number of dimensions.
In particular, in contrast to `AbstractArray`, scalar dimensions are not discarded:

```julia

```

### VectorInterface.jl

As part of the `TensorKit` interface, `AbstractBlockTensorMap` also implements `VectorInterface`.
This means that you can efficiently add, scale, and compute the inner product of `AbstractBlockTensorMap` objects.

```julia

```

### TensorOperations.jl

The `TensorOperations.jl` interface is also implemented for `AbstractBlockTensorMap`.
In particular, the `AbstractBlockTensorMap` can be contracted with other `AbstractBlockTensorMap` objects, as well as with `AbstractTensorMap` objects.
In order for that mix to work, the `AbstractTensorMap` objects are automatically converted to `AbstractBlockTensorMap` objects with a single tensor, i.e. the sum spaces will be a sum of one space.

```julia

```

### Factorizations

Currently, there is only rudimentary support for factorizations of `AbstractBlockTensorMap` objects.
In particular, the implementations are not yet optimized for performance, and the factorizations are typically carried out by mapping to a dense tensor, and then performing the factorization on that tensor.

```julia

```

!!!note Most factorizations do not need to retain the additional imposed block structure. In particular, constructions of orthogonal bases will typically mix up the subspaces, and as such the resulting vector spaces will be `SumSpace`s of a single term.
