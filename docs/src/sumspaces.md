## Direct Sum Spaces

The underlying concept that defines any array (or operator) that has some blocked structure is that of a direct sum of vector spaces.
These spaces are a natural extension of the `TensorKit` vector spaces, and you can think of them as a way to lazily concatenate multiple vector spaces into one.

### `SumSpace`

In `BlockTensorKit`, we provide a type `SumSpace` that allows you to define such direct sums.
They can be defined either directly via the constructor, or by using the `⊕` operator.

```@example sumspaces

```

Essentially, that is all there is to it, and you can now use these `SumSpace` objects much in the same way as you would use an `IndexSpace` object in `TensorKit`.
In particular, it adheres to the interface of `ElementarySpace`, which means that you can query the properties as you would expect.

```@example sumspaces

```

The main difference is that the object retains the information about the individual spaces, and you can query them by indexing into the object.

```@example sumspaces

```

### `ProductSumSpace` and `TensorMapSumSpace`

Because these objects are naturally `ElementarySpace` objects, they can be used in the construction of `ProductSpace` and `HomSpace` objects, and in particular, they can be used to define the spaces of `TensorMap` objects.

```@example sumspaces

```

### `SumSpaceIndices`

Finally, since the `SumSpace` object is the underlying structure of a blocked tensor, it can be convenient to have a way to obtain the vector spaces of the constituent parts.
For this, we provide the `SumSpaceIndices` object, which can be used to efficiently iterate over the indices of the individual spaces.
In particular, we expose the `eachspace` function, similar to `eachindex`, to obtain such an iterator.

```@example sumspaces

```