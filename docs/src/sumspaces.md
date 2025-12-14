## [Direct Sum Spaces](@id sec_sumspaces)

The underlying concept that defines any array (or operator) that has some blocked structure is that of a direct sum of vector spaces.
These spaces are a natural extension of the `TensorKit` vector spaces, and you can think of them as a way to lazily concatenate multiple vector spaces into one.

### `SumSpace`

In `BlockTensorKit`, we provide a type `SumSpace` that allows you to define such direct sums.
They can be defined either directly via the constructor, or by using the `⊞` (`\boxplus<TAB>`) operator.
In order for the direct sum to be wll-defined, all components must have the same value of `isdual`.

Essentially, that is all there is to it, and you can now use these `SumSpace` objects much in the same way as you would use an `IndexSpace` object in `TensorKit`.
In particular, it adheres to the interface of `ElementarySpace`, which means that you can query the properties as you would expect.

!!! note

    The notion of a direct sum of vector spaces is used in both TensorKit (`⊕` or `oplus`) and BlockTensorKit (`⊞` or `boxplus`).
    Both functions achieve almost the same thing, and `BlockTensorKit.⊞` can be thought of as a _lazy_ version of `TensorKit.⊕`.

```@repl sumspaces
using TensorKit, BlockTensorKit
V = ℂ^1 ⊞ ℂ^2 ⊞ ℂ^3
ℂ^2 ⊞ (ℂ^2)' ⊞ ℂ^2 # error
dim(V)
isdual(V)
isdual(V')
field(V)
spacetype(V)
InnerProductStyle(V)
```

The main difference is that the object retains the information about the individual spaces, and you can query them by indexing into the object.

```@repl sumspaces
length(V)
V[1]
```

### `ProductSumSpace` and `TensorMapSumSpace`

Because these objects are naturally `ElementarySpace` objects, they can be used in the construction of `ProductSpace` and `HomSpace` objects, and in particular, they can be used to define the spaces of `TensorMap` objects.
Additionally, when mixing spaces and their sumspaces, all components are promoted to `SumSpace` instances.

```@repl sumspaces
V1 = ℂ^1 ⊞ ℂ^2 ⊞ ℂ^3
V2 = ℂ^2
V1 ⊗ V2 ⊗ V1' == V1 * V2 * V1' == ProductSpace(V1,V2,V1') == ProductSpace(V1,V2) ⊗ V1'
V1^3
dim(V1 ⊗ V2)
dims(V1 ⊗ V2)
dual(V1 ⊗ V2)
spacetype(V1 ⊗ V2)
spacetype(typeof(V1 ⊗ V2))
```

```@repl sumspaces
W = V1 → V2
field(W)
dual(W)
adjoint(W)
spacetype(W)
spacetype(typeof(W))
W[1]
W[2]
dim(W)
```

### `SumSpaceIndices`

Finally, since the `SumSpace` object is the underlying structure of a blocked tensor, it can be convenient to have a way to obtain the vector spaces of the constituent parts.
For this, we provide the `SumSpaceIndices` object, which can be used to efficiently iterate over the indices of the individual spaces.
In particular, we expose the `eachspace` function, similar to `eachindex`, to obtain such an iterator.

```@repl sumspaces
W = V1 * V2 → V2 * V1
eachspace(W)
```
