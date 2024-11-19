# BlockTensorKit.jl

*A Julia package for handling arrays-of-tensors, built on top of [TensorKit.jl](https://github.com/Jutho/TensorKit.jl)*

```@meta
CurrentModule = TensorKit
```

## Package summary

In the context of developing efficient tensor network algorithms, it can sometimes be convenient to write a tensor as a concatenation of other tensors, without explicitly merging them.
This is helpful whenever there are some guarantees on the resulting structure, such as sparsity patterns, triangular structures, or just as a way of keeping things organized.
One particular example, for which this package is primarily developed, is the construction of Matrix Product Operators (MPOs) that represent a sum of local operators, both on 1-dimensional geometries, but also for more general tree-like geometries.
In those cases, the combination of an upper-triangular blocked structure, as well as efficient usage of the sparsity, can not only greatly speed up runtime, but also facilitates rapid development of novel algorithms.

Mathematically speaking, we can consider these blocked tensors as acting on direct sums of vector spaces, where the indiviual vector spaces are supplied by TensorKit.
This leads to a very natural generalization of `AbstractTensorMap`, which is able to handle arbitrary symmetries.

BlockTensorKit.jl aims to provide a convenient interface to such blocked tensors.
In particular, the central types of this package (`<:AbstractBlockTensorMap`) could be describes as having both `AbstractArray`-like interfaces, which allow indexing as well as slicing operations, and `AbstractTensorMap`-like interfaces, allowing linear algebra routines, tensor contraction and tensor factorization.
The goal is to abstract away the need to deal with the inner structures of such tensors as much as possible, and have the ability to replace `AbstractTensorMap`s with `AbstractBlockTensorMap` without having to change the high-level code.

As these kinds of operations typically appear in performance-critical sections of the code, computational efficiency and performance are high on the priority list.
As such, a secondary aim of this package is to provide different algorithms that enable maximal usage of sparsity, multithreading, and other tricks to obtain close-to-maximal performance.

## Contents of the manual

The manual fort his package is separated into 4 large parts.
The first part focusses on the spacetype that underlies these tensors, which contain the necessary information to construct them.
This is followed by a section on `BlockTensorMap`, highlighting the capabilities and interface.
Then, we elaborate on `SparseBlockTensorMap`, which contains the sparse variant.
Finally, we collect all docstrings.

```@contents
Pages = ["sumspaces.md", "blocktensor.md", "sparseblocktensor.md", "lib.md"]
```