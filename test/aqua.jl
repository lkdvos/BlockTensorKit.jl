using Aqua: Aqua
using BlockTensorKit: BlockTensorKit
# dont test for piracies right now, these are deliberate.
Aqua.test_all(BlockTensorKit; piracies=false)
