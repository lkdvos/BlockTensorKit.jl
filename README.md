<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/lkdvos/BlockTensorKit.jl/blob/main/docs/src/assets/logo.svg">
    <img alt="BlockTensorKit.jl logo" src="https://github.com/lkdvos/BlockTensorKit.jl/blob/main/docs/src/assets/logo.svg" width="150">
</picture>

# BlockTensorKit

*A Julia package for handling arrays-of-tensors, built on top of [TensorKit.jl](https://github.com/Jutho/TensorKit.jl)*

| **Documentation** | **Build Status** | **Coverage** | **Quality assurance** |
|:-----------------:|:----------------:|:------------:|:---------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] |


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lkdvos.github.io/BlockTensorKit.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lkdvos.github.io/BlockTensorKit.jl/dev

[ci-img]: https://github.com/lkdvos/BlockTensorKit.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/lkdvos/BlockTensorKit.jl/actions/workflows/CI.yml

[codecov-img]: https://codecov.io/gh/lkdvos/BlockTensorKit.jl/graph/badge.svg?token=C1QPCRT1NT
[codecov-url]: https://codecov.io/gh/lkdvos/BlockTensorKit.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl


In the context of developing efficient tensor network algorithms, it can sometimes be convenient to write a tensor as a concatenation of other tensors, without explicitly merging them.
This is helpful whenever there are some guarantees on the resulting structure, such as sparsity patterns, triangular structures, or just as a way of keeping things organized.
One particular example, for which this package is primarily developed, is the construction of Matrix Product Operators (MPOs) that represent a sum of local operators, both on 1-dimensional geometries, but also for more general tree-like geometries.
In those cases, the combination of an upper-triangular blocked structure, as well as efficient usage of the sparsity, can not only greatly speed up runtime, but also facilitates rapid development of novel algorithms.

Mathematically speaking, we can consider these blocked tensors as acting on direct sums of vector spaces, where the indiviual vector spaces are supplied by TensorKit.
This leads to a very natural generalization of `AbstractTensorMap`, which is able to handle arbitrary symmetries.

BlockTensorKit.jl aims to provide a convenient interface to such blocked tensors.
In particular, the central types of this package (`<:AbstractBlockTensorMap`) could be describes as having both `AbstractArray`-like interfaces, which allow indexing as well as slicing operations, and `AbstractTensorMap`-like interfaces, allowing linear algebra routines, tensor contraction and tensor factorization.
The goal is to abstract away the need to deal with the inner structures of such tensors as much as possible, and have the ability to replace `AbstractTensorMap`s with `AbstractBlockTensorMap` without having to change the high-level code.

For examples and further information, please check out the [documentation](https://lkdvos.github.io/BlockTensorKit.jl/dev).
