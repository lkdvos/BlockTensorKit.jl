using SafeTestsets

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = uppercase(
    if isnothing(arg_id)
        uppercase(get(ENV, "GROUP", "ALL"))
    else
        uppercase(only(match(pat, ARGS[arg_id]).captures))
    end,
)

@time begin
    if GROUP == "ALL" || GROUP == "VECTORSPACES"
        @time @safetestset "SumSpace" begin
            include("sumspace.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "ABSTRACTTENSOR"
        @time @safetestset "Indexing" begin
            include("indexing.jl")
        end
        @time @safetestset "BlockTensor" begin
            include("blocktensor.jl")
        end
        @time @safetestset "SparseBlockTensor" begin
            include("sparseblocktensor.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "LINALG"
        @time @safetestset "VectorInterface" begin
            include("vectorinterface.jl")
        end
        @time @safetestset "indexmanipulations" begin
            include("indexmanipulations.jl")
        end
        @time @safetestset "TensorOperations" begin
            include("tensoroperations.jl")
        end
        @time @safetestset "factorizations" begin
            include("factorizations.jl")
        end
    end
end
