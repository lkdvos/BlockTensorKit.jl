using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "VectorSpaces"
       @time @safetestset "SumSpace" begin include("sumspace.jl") end 
    end
    
    if GROUP == "All" || GROUP == "TensorOperations"
       @time @safetestset "TensorOperations" begin include("tensoroperations.jl") end 
    end
end
