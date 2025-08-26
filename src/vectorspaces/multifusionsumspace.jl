# custom methods for SumSpaces

function Base.oneunit(S::SumSpace{Vect[IsingBimodule]})
    @assert !isempty(S) "Cannot determine type of empty space"
    allequal(a.row for a in sectors(S)) && allequal(a.col for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S are not all equal"))
    first(sectors(S)).row == first(sectors(S)).col || throw(ArgumentError("non-diagonal SumSpace $S"))
    return SumSpace(oneunit(first(S.spaces)))
end

function TensorKit.rightoneunit(S::SumSpace{Vect[IsingBimodule]})
    @assert !isempty(S) "Cannot determine type of empty space"
    allequal(a.col for a in sectors(S)) || throw(ArgumentError("sectors of $S do not have the same rightone"))
    return SumSpace(TensorKit.rightoneunit(first(S.spaces)))
end

function TensorKit.leftoneunit(S::SumSpace{Vect[IsingBimodule]})
    @assert !isempty(S) "Cannot determine type of empty space"
    allequal(a.row for a in sectors(S)) || throw(ArgumentError("sectors of $S do not have the same leftone"))
    return SumSpace(TensorKit.leftoneunit(first(S.spaces)))
end

function TensorKit.blocksectors(W::TensorMapSpace{SumSpace{Vect[IsingBimodule]},N₁,N₂}) where {N₁,N₂}
    codom = codomain(W)
    dom = domain(W)
    if N₁ == 0 && N₂ == 0
        return [IsingBimodule(1, 1, 0), IsingBimodule(2, 2, 0)]
    elseif N₁ == 0
        @assert N₂ != 0 "one of Type IsingBimodule doesn't exist"
        return filter!(isone, collect(blocksectors(dom)))
    elseif N₂ == 0
        @assert N₁ != 0 "one of Type IsingBimodule doesn't exist"
        return filter!(isone, collect(blocksectors(codom)))
    elseif N₂ <= N₁ # keep intersection
        return filter!(c -> TK.hasblock(codom, c), collect(blocksectors(dom)))
    else
        return filter!(c -> TK.hasblock(dom, c), collect(blocksectors(codom)))
    end
end

function TensorKit.blocksectors(P::ProductSpace{SumSpace{Vect[IsingBimodule]},N}) where {N}
    I = sectortype(P) # IsingBimodule
    bs = Vector{I}()
    if N == 0
        return [IsingBimodule(1, 1, 0), IsingBimodule(2, 2, 0)]
    elseif N == 1
        for s in sectors(P)
            push!(bs, first(s))
        end
    else
        for s in sectors(P)
            for c in ⊗(s...)
                if !(c in bs)
                    push!(bs, c)
                end
            end
        end
    end
    return sort!(bs)
end