using Documenter
using BlockTensorKit

pages = [
    "Home" => "index.md",
    "Manual" => ["SumSpace" => "sumspaces.md", "BlockTensors" => "blocktensors.md"],
    "Library" => "lib.md",
]

makedocs(;
    modules = [BlockTensorKit],
    sitename = "BlockTensorKit.jl",
    authors = "Lukas Devos",
    warnonly = [:missing_docs, :cross_references],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(),
        repolink = "https://github.com/lkdvos/BlockTensorKit.jl.git",
    ),
    pages = pages,
    pagesonly = true,
    repo = "github.com/lkdvos/BlockTensorKit.jl.git",
)

deploydocs(; repo = "github.com/lkdvos/BlockTensorKit.jl.git", push_preview = true)
