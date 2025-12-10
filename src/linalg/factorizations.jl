using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, YALAPACK.BlasMat, Algorithm
import MatrixAlgebraKit as MAK

# Type piracy for defining the MAK rules on BlockArrays!
# -----------------------------------------------------

const BlockBlasMat{T <: MAK.BlasFloat} = BlockMatrix{T}

function MatrixAlgebraKit.one!(A::BlockBlasMat)
    _one, _zero = one(eltype(A)), zero(eltype(A))
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        A[i, j] = ifelse(i == j, _one, _zero)
    end
    return A
end

for f in
    [
        :svd_compact, :svd_full, :svd_vals,
        :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null,
        :eig_full, :eig_vals, :eigh_full, :eigh_vals,
        :left_polar, :right_polar,
        :project_hermitian, :project_antihermitian, :project_isometric,
    ]
    f! = Symbol(f, :!)
    @eval MAK.default_algorithm(::typeof($f!), ::Type{T}; kwargs...) where {T <: AbstractBlockTensorMap} =
        MAK.default_algorithm($f!, eltype(T); kwargs...)
end

for f! in (
        :qr_compact!, :qr_full!, :lq_compact!, :lq_full!,
        :eig_full!, :eigh_full!, :svd_compact!, :svd_full!,
        :left_polar!, :right_polar!,
    )
    @eval function MAK.$f!(t::AbstractBlockTensorMap, F, alg::AbstractAlgorithm)
        TensorKit.foreachblock(t, F...) do _, (tblock, Fblocks...)
            Fblocks′ = MAK.$f!(Array(tblock), alg)
            # deal with the case where the output is not in-place
            for (b′, b) in zip(Fblocks′, Fblocks)
                b === b′ || copy!(b, b′)
            end
            return nothing
        end
        return F
    end
end

# Handle these separately because single output instead of tuple
for f! in (
        :qr_null!, :lq_null!,
        :svd_vals!, :eig_vals!, :eigh_vals!,
        :project_hermitian!, :project_antihermitian!, :project_isometric!,
    )
    @eval function MAK.$f!(t::AbstractBlockTensorMap, N, alg::AbstractAlgorithm)
        TensorKit.foreachblock(t, N) do _, (tblock, Nblock)
            Nblock′ = MAK.$f!(Array(tblock), alg)
            # deal with the case where the output is not the same as the input
            Nblock === Nblock′ || copy!(Nblock, Nblock′)
            return nothing
        end
        return N
    end
end

for f in (
        :svd_compact, :svd_full, :svd_vals, :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null, :eig_full, :eig_vals, :eigh_full,
        :eigh_vals, :left_polar, :right_polar,
    )
    f! = Symbol(f, :!)
    @eval MAK.$f!(t::BlockBlasMat, F, alg::AbstractAlgorithm) = $f!(Array(t), alg)
    @eval MAK.$f!(t::BlockBlasMat, F, alg::MAK.DiagonalAlgorithm) = error("Not diagonal")
end

# specializations until fixes in base package
function MAK.is_left_isometric(A::BlockMatrix; atol::Real = 0, rtol::Real = MAK.defaulttol(A), norm = LinearAlgebra.norm)
    P = A' * A
    nP = norm(P) # isapprox would use `rtol * max(norm(P), norm(I))`
    for I in MAK.diagind(P)
        P[I] -= 1
    end
    return norm(P) <= max(atol, rtol * nP) # assume that the norm of I is `sqrt(n)`
end
function MAK.is_right_isometric(A::BlockMatrix; atol::Real = 0, rtol::Real = MAK.defaulttol(A), norm = LinearAlgebra.norm)
    P = A * A'
    nP = norm(P) # isapprox would use `rtol * max(norm(P), norm(I))`
    for I in MAK.diagind(P)
        P[I] -= 1
    end
    return norm(P) <= max(atol, rtol * nP) # assume that the norm of I is `sqrt(n)`
end

# disambiguations
# for (f!, Alg) in (
#         (:lq_compact!, :LAPACK_HouseholderLQ), (:lq_full!, :LAPACK_HouseholderLQ), (:lq_null!, :LAPACK_HouseholderLQ),
#         (:lq_compact!, :LQViaTransposedQR), (:lq_full!, :LQViaTransposedQR), (:lq_null!, :LQViaTransposedQR),
#         (:qr_compact!, :LAPACK_HouseholderQR), (:qr_full!, :LAPACK_HouseholderQR), (:qr_null!, :LAPACK_HouseholderQR),
#         (:svd_compact!, :LAPACK_SVDAlgorithm), (:svd_full!, :LAPACK_SVDAlgorithm), (:svd_vals!, :LAPACK_SVDAlgorithm),
#         (:eig_full!, :LAPACK_EigAlgorithm), (:eig_trunc!, :TruncatedAlgorithm), (:eig_vals!, :LAPACK_EigAlgorithm),
#         (:eigh_full!, :LAPACK_EighAlgorithm), (:eigh_trunc!, :TruncatedAlgorithm), (:eigh_vals!, :LAPACK_EighAlgorithm),
#         (:left_polar!, :PolarViaSVD), (:right_polar!, :PolarViaSVD),
#     )
#     @eval MAK.$f!(t::BlockBlasMat, F, alg::MAK.$Alg) = $f!(Array(t), alg)
# end
#
# const GPU_QRAlgorithm = Union{MAK.CUSOLVER_HouseholderQR, MAK.ROCSOLVER_HouseholderQR}
# for f! in (:qr_compact!, :qr_full!, :qr_null!)
#     @eval MAK.$f!(t::BlockBlasMat, QR, alg::GPU_QRAlgorithm) = error()
# end
#
# for (f!, Alg) in (
#         (:eigh_full!, :GPU_EighAlgorithm), (:eigh_vals!, :GPU_EighAlgorithm),
#         (:eig_full!, :GPU_EigAlgorithm), (:eig_vals!, :GPU_EigAlgorithm),
#         (:svd_full!, :GPU_SVDAlgorithm), (:svd_compact!, :GPU_SVDAlgorithm), (:svd_vals!, :GPU_SVDAlgorithm),
#     )
#     @eval MAK.$f!(t::BlockBlasMat, F, alg::MAK.$Alg) = error()
# end


# for f in (:qr, :lq, :eig, :eigh, :gen_eig, :svd, :polar)
#     default_f_algorithm = Symbol(:default_, f, :_algorithm)
#     @eval MAK.$default_f_algorithm(::Type{<:BlockBlasMat{T}}; kwargs...) where {T} =
#         MAK.$default_f_algorithm(Matrix{T}; kwargs...)
# end

# Make sure sparse blocktensormaps have dense outputs
function MAK.initialize_output(::typeof(qr_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = ⊕(fuse(codomain(t)))
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
    V_Q = ⊕(fuse(domain(t)))
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

function MAK.initialize_output(::typeof(eigh_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = ⊕(fuse(domain(t)))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = dense_similar(t, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(eig_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = ⊕(fuse(domain(t)))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = dense_similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(svd_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = ⊕(fuse(codomain(t)))
    V_dom = ⊕(fuse(domain(t)))
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
