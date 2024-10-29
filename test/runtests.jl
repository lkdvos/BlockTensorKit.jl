using SafeTestsets

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = if isnothing(arg_id)
    uppercase(get(ENV, "GROUP", "ALL"))
else
    uppercase(only(match(pat, ARGS[arg_id]).captures))
end |> uppercase

@time begin
    if GROUP == "ALL" || GROUP == "VECTORSPACES"
       @time @safetestset "SumSpace" begin include("sumspace.jl") end 
    end
    
    if GROUP == "ALL" || GROUP == "ABSTRACTARRAY"
        @time @safetestset "Indexing" begin include("indexing.jl") end
    end

    if GROUP == "ALL" || GROUP == "VECTORINTERFACE"
        @time @safetestset "VectorInterface" begin include("vectorinterface.jl") end
    end

    if GROUP == "ALL" || GROUP == "TENSOROPERATIONS"
       @time @safetestset "TensorOperations" begin include("tensoroperations.jl") end 
    end
end
