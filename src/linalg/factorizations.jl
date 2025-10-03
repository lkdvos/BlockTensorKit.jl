using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, YALAPACK.BlasMat, Algorithm
import MatrixAlgebraKit as MAK

# Type piracy for defining the MAK rules on BlockArrays!
# -----------------------------------------------------

MAK.@algdef BlockAlgorithm

const BlockBlasMat{T <: MAK.BlasFloat} = BlockMatrix{T}

for f in (
        :svd_compact, :svd_full, :svd_trunc, :svd_vals, :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null, :eig_full, :eig_trunc, :eig_vals, :eigh_full,
        :eigh_trunc, :eigh_vals, :left_polar, :right_polar,
    )
    f! = Symbol(f, :!)

    @eval MAK.copy_input(::typeof($f), A::BlockBlasMat) = Array(A)
    @eval MAK.$f!(t::BlockBlasMat, F, alg::BlockAlgorithm) = $f(t; alg.kwargs...)
end

for f in (:qr, :lq, :eig, :eigh, :gen_eig, :svd, :polar)
    default_f_algorithm = Symbol(:default_, f, :_algorithm)
    @eval MAK.$default_f_algorithm(::Type{<:BlockBlasMat{T}}; kwargs...) where {T} =
        BlockAlgorithm(; kwargs...)
end

# Make sure sparse blocktensormaps have dense outputs
function MAK.initialize_output(::typeof(qr_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(codomain(t))
    Q = dense_similar(t, codomain(t) ← V_Q)
    R = dense_similar(t, V_Q ← domain(t))
    return Q, R
end
function MAK.initialize_output(::typeof(qr_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    Q = dense_similar(t, codomain(t) ← V_Q)
    R = dense_similar(t, V_Q ← domain(t))
    return Q, R
end
function MAK.initialize_output(::typeof(qr_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = dense_similar(t, codomain(t) ← V_N)
    return N
end

function MAK.initialize_output(::typeof(lq_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(domain(t))
    L = dense_similar(t, codomain(t) ← V_Q)
    Q = dense_similar(t, V_Q ← domain(t))
    return L, Q
end
function MAK.initialize_output(::typeof(lq_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    L = dense_similar(t, codomain(t) ← V_Q)
    Q = dense_similar(t, V_Q ← domain(t))
    return L, Q
end
function MAK.initialize_output(::typeof(lq_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    N = dense_similar(t, V_N ← domain(t))
    return N
end

function MAK.initialize_output(::typeof(left_polar!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    W = dense_similar(t, space(t))
    P = dense_similar(t, domain(t) ← domain(t))
    return W, P
end
function MAK.initialize_output(::typeof(right_polar!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    P = dense_similar(t, codomain(t) ← codomain(t))
    Wᴴ = dense_similar(t, space(t))
    return P, Wᴴ
end

function MAK.initialize_output(::typeof(left_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    return dense_similar(t, codomain(t) ← V_N)
end
function MAK.initialize_output(::typeof(right_null!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    return dense_similar(t, V_N ← domain(t))
end

function MAK.initialize_output(::typeof(eigh_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = dense_similar(t, codomain(t) ← V_D)
    return D, V
end
function MAK.initialize_output(::typeof(eig_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = dense_similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(svd_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = dense_similar(t, codomain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = dense_similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end
function MAK.initialize_output(::typeof(svd_compact!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = dense_similar(t, codomain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
    Vᴴ = dense_similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

# Disambiguate Diagonal implementations
# -------------------------------------
# these shouldn't ever happen as blocktensors aren't diagonal
for f! in (:eig_full!, :eigh_full!, :lq_compact!, :lq_full!, :qr_compact!, :qr_full!, :svd_full!)
    @eval function MAK.initialize_output(::typeof($f!), t::AbstractBlockTensorMap, alg::DiagonalAlgorithm)
        error("Blocktensors are incompatible with diagonal algorithm")
    end
end
