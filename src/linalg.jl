function LinearAlgebra.mul!(C::BlockTensorMap, A::BlockTensorMap, α::Number)
    space(C) == space(A) || throw(SpaceMismatch())
    mul!.(parent(C), parent(A), α)
    return C
end

function LinearAlgebra.mul!(C::BlockTensorMap, α::Number, A::BlockTensorMap)
    space(C) == space(A) || throw(SpaceMismatch())
    mul!.(parent(C), α, parent(A))
    return C
end

function LinearAlgebra.rmul!(t::BlockTensorMap, α::Number)
    rmul!.(parent(t), α)
    return t
end
function LinearAlgebra.lmul!(α::Number, t::BlockTensorMap)
    lmul!.(α, parent(t))
    return t
end

function LinearAlgebra.axpy!(α::Number, t1::BlockTensorMap, t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    axpy!.(α, parent(t1), parent(t2))
    return t2
end

function LinearAlgebra.axpby!(α::Number, t1::BlockTensorMap, β::Number, t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    axpby!.(α, parent(t1), β, parent(t2))
    return t2
end

function LinearAlgebra.dot(t1::BlockTensorMap, t2::BlockTensorMap)
    space(t1) == space(t2) || throw(SpaceMismatch())
    return sum(zip(parent(t1), parent(t2))) do (x, y)
        return dot(x, y)
    end
end

function LinearAlgebra.mul!(tC::BlockTensorMap, tA::BlockTensorMap, tB::BlockTensorMap,
                            α=true, β=false)
    pC = (ntuple(identity, numout(tC)), ntuple(identity, numin(tC)) .+ numout(tC))
    pA = (ntuple(identity, numout(tA)), ntuple(identity, numin(tA)) .+ numout(tA))
    pB = (ntuple(identity, numout(tB)), ntuple(identity, numin(tB)) .+ numout(tB))

    return tensorcontract!(tC, pC, tA, pA, :N, tB, pB, :N, α, β)
end

function LinearAlgebra.norm(tA::BlockTensorMap, p::Real=2)
    return LinearAlgebra.norm(parent(tA), p)
end

function TensorKit.:⊗(t1::BlockTensorMap{S}, t2::BlockTensorMap{S}) where {S}
    pC = ((codomainind(t1)..., (codomainind(t2) .+ numind(t1))...),
          (domainind(t1)..., (domainind(t2) .+ numind(t1))...))
    pA = (allind(t1), ())
    pB = ((), allind(t2))
    return tensorproduct(pC, t1, pA, :N, t2, pB, :N)
end

#===========================================================================================
    Inverses
===========================================================================================#

# function Base.inv(t::BlockTensorMap)
#     cod = codomain(t)
#     dom = domain(t)
#     isisomorphic(cod, dom) ||
#         throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))

#     tdst = TensorMap(data, dom ← cod)
#     for (c, b) in blocks(t)
#         blocks(tdst)[c] = inv(b)
#     end
#     return tdst
# end
# function LinearAlgebra.pinv(t::BlockTensorMap; kwargs...)
#     cod = codomain(t)
#     dom = domain(t)
#     isisomorphic(cod, dom) ||
#         throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))

#     tdst = TensorMap(data, dom ← cod)
#     for (c, b) in blocks(t)
#         blocks(tdst)[c] = pinv(b; kwargs...)
#     end
#     return tdst
# end

function Base.:(\)(t1::BlockTensorMap, t2::BlockTensorMap)
    codomain(t1) == codomain(t2) ||
        throw(SpaceMismatch("non-matching codomains in t1 \\ t2"))

    tdst = TensorMap(undef, promote_type(scalartype(t1), scalartype(t2)), domain(t1),
                     domain(t2))
    for (c, b) in blocks(t1)
        result = b \ block(t2, c)

        blocks(tdst)[c] = b \ block(t2, c)
    end

    if sectortype(t1) === Trivial
        data = block(t1, Trivial()) \ block(t2, Trivial())
        return TensorMap(data, domain(t1) ← domain(t2))
    else
        cod = codomain(t1)
        data = SectorDict(c => block(t1, c) \ block(t2, c)
                          for c in blocksectors(codomain(t1)))
        return TensorMap(data, domain(t1) ← domain(t2))
    end
end
function Base.:(/)(t1::BlockTensorMap, t2::BlockTensorMap)
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("non-matching domains in t1 / t2"))
    if sectortype(t1) === Trivial
        data = Array(block(t1, Trivial()) / block(t2, Trivial()))
        return TensorMap(data, codomain(t1) ← codomain(t2))
    else
        data = SectorDict(c => Array(block(t1, c) / block(t2, c))
                          for c in blocksectors(domain(t1)))
        return TensorMap(data, codomain(t1) ← codomain(t2))
    end
end
