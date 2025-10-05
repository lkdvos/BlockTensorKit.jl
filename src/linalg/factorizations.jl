using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, YALAPACK.BlasMat, Algorithm
import MatrixAlgebraKit as MAK
using TensorKit.Factorizations: @check_space, @check_scalar

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

for f in (
        :svd_compact, :svd_full, :svd_vals, :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null, :eig_full, :eig_vals, :eigh_full,
        :eigh_vals, :left_polar, :right_polar,
    )
    f! = Symbol(f, :!)
    @eval MAK.$f!(t::BlockBlasMat, F, alg::AbstractAlgorithm) = $f!(Array(t), alg)
    @eval MAK.$f!(t::BlockBlasMat, F, alg::MAK.DiagonalAlgorithm) = error("Not diagonal")
end

# disambiguations
for (f!, Alg) in (
        (:lq_compact!, :LAPACK_HouseholderLQ), (:lq_full!, :LAPACK_HouseholderLQ), (:lq_null!, :LAPACK_HouseholderLQ),
        (:lq_compact!, :LQViaTransposedQR), (:lq_full!, :LQViaTransposedQR), (:lq_null!, :LQViaTransposedQR),
        (:qr_compact!, :LAPACK_HouseholderQR), (:qr_full!, :LAPACK_HouseholderQR), (:qr_null!, :LAPACK_HouseholderQR),
        (:svd_compact!, :LAPACK_SVDAlgorithm), (:svd_full!, :LAPACK_SVDAlgorithm), (:svd_vals!, :LAPACK_SVDAlgorithm),
        (:eig_full!, :LAPACK_EigAlgorithm), (:eig_trunc!, :TruncatedAlgorithm), (:eig_vals!, :LAPACK_EigAlgorithm),
        (:eigh_full!, :LAPACK_EighAlgorithm), (:eigh_trunc!, :TruncatedAlgorithm), (:eigh_vals!, :LAPACK_EighAlgorithm),
        (:left_polar!, :PolarViaSVD), (:right_polar!, :PolarViaSVD),
    )
    @eval MAK.$f!(t::BlockBlasMat, F, alg::MAK.$Alg) = $f!(Array(t), alg)
end

const GPU_QRAlgorithm = Union{MAK.CUSOLVER_HouseholderQR, MAK.ROCSOLVER_HouseholderQR}
for f! in (:qr_compact!, :qr_full!, :qr_null!)
    @eval MAK.$f!(t::BlockBlasMat, QR, alg::GPU_QRAlgorithm) = error()
end

for (f!, Alg) in (
        (:eigh_full!, :GPU_EighAlgorithm), (:eigh_vals!, :GPU_EighAlgorithm),
        (:eig_full!, :GPU_EigAlgorithm), (:eig_vals!, :GPU_EigAlgorithm),
        (:svd_full!, :GPU_SVDAlgorithm), (:svd_compact!, :GPU_SVDAlgorithm), (:svd_vals!, :GPU_SVDAlgorithm),
    )
    @eval MAK.$f!(t::BlockBlasMat, F, alg::MAK.$Alg) = error()
end


for f in (:qr, :lq, :eig, :eigh, :gen_eig, :svd, :polar)
    default_f_algorithm = Symbol(:default_, f, :_algorithm)
    @eval MAK.$default_f_algorithm(::Type{<:BlockBlasMat{T}}; kwargs...) where {T} =
        MAK.$default_f_algorithm(Matrix{T}; kwargs...)
end

# Make sure sparse blocktensormaps have dense outputs
function MAK.check_input(::typeof(qr_full!), t::AbstractBlockTensorMap, QR, ::AbstractAlgorithm)
    Q, R = QR

    # type checks
    @assert Q isa AbstractTensorMap
    @assert R isa AbstractTensorMap

    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = TK.oplus(fuse(codomain(t)))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end
MAK.check_input(::typeof(qr_full!), t::AbstractBlockTensorMap, QR, ::DiagonalAlgorithm) = error()

function MAK.initialize_output(::typeof(qr_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = TK.oplus(fuse(codomain(t)))
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

function MAK.check_input(::typeof(lq_full!), t::AbstractBlockTensorMap, LQ, ::AbstractAlgorithm)
    L, Q = LQ

    # type checks
    @assert L isa AbstractTensorMap
    @assert Q isa AbstractTensorMap

    # scalartype checks
    @check_scalar L t
    @check_scalar Q t

    # space checks
    V_Q = TK.oplus(fuse(domain(t)))
    @check_space(L, codomain(t) ← V_Q)
    @check_space(Q, V_Q ← domain(t))

    return nothing
end
MAK.check_input(::typeof(lq_full!), t::AbstractBlockTensorMap, LQ, ::DiagonalAlgorithm) = error()

function MAK.initialize_output(::typeof(lq_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_Q = TK.oplus(fuse(domain(t)))
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

function MAK.check_input(::typeof(MAK.left_orth_polar!), t::AbstractBlockTensorMap, WP, ::AbstractAlgorithm)
    codomain(t) ≿ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `codomain(t) ≿ domain(t)`"))

    W, P = WP
    @assert W isa AbstractTensorMap
    @assert P isa AbstractTensorMap

    # scalartype checks
    @check_scalar W t
    @check_scalar P t

    # space checks
    VW = TK.oplus(fuse(domain(t)))
    @check_space(W, codomain(t) ← VW)
    @check_space(P, VW ← domain(t))

    return nothing
end
function MAK.initialize_output(::typeof(left_polar!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    W = dense_similar(t, space(t))
    P = dense_similar(t, domain(t) ← domain(t))
    return W, P
end

function MAK.check_input(::typeof(MAK.right_orth_polar!), t::AbstractBlockTensorMap, PWᴴ, ::AbstractAlgorithm)
    codomain(t) ≾ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `domain(t) ≿ codomain(t)`"))

    P, Wᴴ = PWᴴ
    @assert P isa AbstractTensorMap
    @assert Wᴴ isa AbstractTensorMap

    # scalartype checks
    @check_scalar P t
    @check_scalar Wᴴ t

    # space checks
    VW = TK.oplus(fuse(codomain(t)))
    @check_space(P, codomain(t) ← VW)
    @check_space(Wᴴ, VW ← domain(t))

    return nothing
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

function MAK.check_input(::typeof(eigh_full!), t::AbstractBlockTensorMap, DV, ::AbstractAlgorithm)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    D, V = DV

    # type checks
    @assert D isa DiagonalTensorMap
    @assert V isa AbstractTensorMap

    # scalartype checks
    @check_scalar D t real
    @check_scalar V t

    # space checks
    V_D = TK.oplus(fuse(domain(t)))
    @check_space(D, V_D ← V_D)
    @check_space(V, codomain(t) ← V_D)

    return nothing
end
MAK.check_input(::typeof(eigh_full!), t::AbstractBlockTensorMap, DV, ::DiagonalAlgorithm) = error()

function MAK.check_input(::typeof(eigh_vals!), t::AbstractBlockTensorMap, D, ::AbstractAlgorithm)
    @check_scalar D t real
    @assert D isa DiagonalTensorMap
    V_D = TK.oplus(fuse(domain(t)))
    @check_space(D, V_D ← V_D)
    return nothing
end
MAK.check_input(::typeof(eigh_vals!), t::AbstractBlockTensorMap, D, ::DiagonalAlgorithm) = error()

function MAK.initialize_output(::typeof(eigh_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = TK.oplus(fuse(domain(t)))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = dense_similar(t, codomain(t) ← V_D)
    return D, V
end


function MAK.check_input(::typeof(eig_full!), t::AbstractBlockTensorMap, DV, ::AbstractAlgorithm)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    D, V = DV

    # type checks
    @assert D isa DiagonalTensorMap
    @assert V isa AbstractTensorMap

    # scalartype checks
    @check_scalar D t complex
    @check_scalar V t complex

    # space checks
    V_D = TK.oplus(fuse(domain(t)))
    @check_space(D, V_D ← V_D)
    @check_space(V, codomain(t) ← V_D)

    return nothing
end
MAK.check_input(::typeof(eig_full!), t::AbstractBlockTensorMap, DV, ::DiagonalAlgorithm) = error()

function MAK.initialize_output(::typeof(eig_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_D = TK.oplus(fuse(domain(t)))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = dense_similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function MAK.check_input(::typeof(svd_full!), t::AbstractBlockTensorMap, USVᴴ, ::AbstractAlgorithm)
    U, S, Vᴴ = USVᴴ

    # type checks
    @assert U isa AbstractTensorMap
    @assert S isa AbstractTensorMap
    @assert Vᴴ isa AbstractTensorMap

    # scalartype checks
    @check_scalar U t
    @check_scalar S t real
    @check_scalar Vᴴ t

    # space checks
    V_cod = TK.oplus(fuse(codomain(t)))
    V_dom = TK.oplus(fuse(domain(t)))
    @check_space(U, codomain(t) ← V_cod)
    @check_space(S, V_cod ← V_dom)
    @check_space(Vᴴ, V_dom ← domain(t))

    return nothing
end
function MAK.initialize_output(::typeof(svd_full!), t::AbstractBlockTensorMap, ::AbstractAlgorithm)
    V_cod = TK.oplus(fuse(codomain(t)))
    V_dom = TK.oplus(fuse(domain(t)))
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
