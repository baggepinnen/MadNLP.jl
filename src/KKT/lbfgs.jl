import LinearAlgebra

@enum(BFGSInitStrategy::Int,
    SCALAR1  = 1,
    SCALAR2  = 2,
    SCALAR3  = 3,
    SCALAR4  = 4,
    CONSTANT = 5,
)

abstract type AbstractQuasiNewton end

mutable struct CompactLBFGS{T, VT, MT} <: AbstractQuasiNewton
    init_strategy::BFGSInitStrategy
    max_mem::Int
    current_mem::Int
    ξ::T
    Sk::MT # n x p
    Yk::MT # n x p
    Lk::MT # p x p
    Dk::VT # p
    Mk::MT # p x p (for Cholesky factorization Mₖ = Jₖᵀ Jₖ)
    Tk::MT # 2p x 2p
    SdotS::MT # p x p
    U::MT # n x 2p
    _Y::MT # p x n
    Utilde::MT # n x 2p
    _w1::VT
    _w2::VT
end

function CompactLBFGS(n::Int; max_mem=5, ξ=1.0, init_strategy=SCALAR1)
    return CompactLBFGS(
        init_strategy,
        max_mem,
        0,
        ξ,
        zeros(n, 0),
        zeros(n, 0),
        zeros(n, 0),
        zeros(0),
        zeros(0, 0),
        zeros(0, 0),
        zeros(0, 0),
        zeros(0, 0),
        zeros(0, 0),
        zeros(0, 0),
        zeros(0),
        zeros(0),
    )
end

Base.size(lbfgs::CompactLBFGS) = (size(lbfgs.Sk, 1), lbfgs.current_mem)

function _resize!(lbfgs::CompactLBFGS)
    n, k = size(lbfgs)

    lbfgs.Lk = zeros(k, k)
    lbfgs.SdotS = zeros(k, k)

    lbfgs.Mk = zeros(k, k)
    lbfgs.Tk = zeros(2*k, 2*k)
    lbfgs.U = zeros(n, 2*k)
    lbfgs.Utilde = zeros(n, 2*k)
    lbfgs._Y = zeros(k, n)
    lbfgs._w1 = zeros(k)
    lbfgs._w2 = zeros(k)
    return
end
# augment / shift
function _update_SY!(lbfgs::CompactLBFGS, s, y)
    if lbfgs.current_mem < lbfgs.max_mem
        lbfgs.current_mem += 1
        lbfgs.Sk = hcat(lbfgs.Sk, s)
        lbfgs.Yk = hcat(lbfgs.Yk, y)
        _resize!(lbfgs)
    else
        k = lbfgs.current_mem
        # Shift
        for i_ in 1:k-1
            lbfgs.Sk[:, i_] .= lbfgs.Sk[:, i_+1]
            lbfgs.Yk[:, i_] .= lbfgs.Yk[:, i_+1]
        end
        lbfgs.Sk[:, k] .= s
        lbfgs.Yk[:, k] .= y
    end
end

function _refresh_D!(lbfgs::CompactLBFGS, sk, yk)
    k = lbfgs.current_mem
    sTy = dot(sk, yk)
    if length(lbfgs.Dk) < lbfgs.max_mem
        push!(lbfgs.Dk, sTy)
    else
        # shift
        for i in 1:k-1
            lbfgs.Dk[i] = lbfgs.Dk[i+1]
        end
        lbfgs.Dk[k] = sTy
    end
end
function _refresh_L!(lbfgs::CompactLBFGS)
    mul!(lbfgs.Lk, lbfgs.Sk', lbfgs.Yk)
    LinearAlgebra.UpperTriangular(lbfgs.Lk) .= 0.0
end
function _refresh_STS!(lbfgs::CompactLBFGS)
    mul!(lbfgs.SdotS, lbfgs.Sk', lbfgs.Sk, 1.0, 1.0)
end

function update_values!(lbfgs::CompactLBFGS, sk, yk)
    if dot(sk, yk) < 1e-8
        return
    end
    _update_SY!(lbfgs, sk, yk)
    # Refresh internal structures
    _refresh_D!(lbfgs, sk, yk)
    _refresh_L!(lbfgs)
    _refresh_STS!(lbfgs)

    k = lbfgs.current_mem

    # Update scalar
    lbfgs.ξ = dot(sk, yk) / dot(sk, sk)

    # TODO: use a buffer for DkLk
    DkLk = (1.0 ./ lbfgs.Dk) .* lbfgs.Lk'
    lbfgs.Mk .= lbfgs.SdotS
    mul!(lbfgs.Mk, lbfgs.Lk, DkLk, 1.0, 1.0)
    # TODO: ensure Mk is symmetric
    lbfgs.Mk .= 0.5 .* (lbfgs.Mk .+ lbfgs.Mk')

    # Factorize as Mₖ = Jₖᵀ Jₖ
    Jₖ = LinearAlgebra.cholesky(lbfgs.Mk)

    # Update low-rank representation
    U1 = view(lbfgs.U, :, 1:k)
    U1ᵀ = lbfgs._Y
    U1ᵀ .= lbfgs.ξ .* lbfgs.Sk'
    mul!(U1ᵀ, DkLk', lbfgs.Yk', 1.0, 1.0)
    ldiv!(Jₖ, U1ᵀ)
    LinearAlgebra.transpose!(U1, U1ᵀ)

    δ = lbfgs._w1
    δ .= .-1.0 ./ sqrt.(lbfgs.Dk)
    U2 = view(lbfgs.U, :, 1+k:2*k)
    U2 .= δ' .* lbfgs.Yk
end


struct LBFGSKKTSystem{T, MT} <: AbstractReducedKKTSystem{T, MT}
    lbfgs::CompactLBFGS
    hess::StrideOneVector{T}
    jac::StrideOneVector{T}
    pr_diag::StrideOneVector{T}
    du_diag::StrideOneVector{T}
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32}
    aug_com::MT
    aug_csc_map::Union{Nothing, Vector{Int}}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32}
    jac_com::MT
    jac_prev::MT
    jac_csc_map::Union{Nothing, Vector{Int}}
    # Info
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    ind_aug_fixed::Vector{Int}
    jacobian_scaling::Vector{T}

    last_grad::Vector{T}
    last_jacl::Vector{T}
    last_x::Vector{T}
end

function LBFGSKKTSystem{T, MT}(
    n::Int, m::Int,
    ind_ineq::Vector{Int}, ind_fixed::Vector{Int},
    lbfgs::CompactLBFGS, jac_sparsity_I, jac_sparsity_J,
) where {T, SpMT, MT}

    # diagonal sparsity
    hess_sparsity_I = collect(1:n)
    hess_sparsity_J = collect(1:n)

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)

    aug_vec_length = n+m
    aug_mat_length = n+m+n_hess+n_jac

    I = Vector{Int32}(undef, aug_mat_length)
    J = Vector{Int32}(undef, aug_mat_length)
    V = Vector{T}(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n+n_jac+n_hess+m

    I[1:n] .= 1:n
    I[n+1:n+n_hess] = hess_sparsity_I
    I[n+n_hess+1:n+n_hess+n_jac] .= (jac_sparsity_I.+n)
    I[n+n_hess+n_jac+1:offset] .= (n+1:n+m)

    J[1:n] .= 1:n
    J[n+1:n+n_hess] = hess_sparsity_J
    J[n+n_hess+1:n+n_hess+n_jac] .= jac_sparsity_J
    J[n+n_hess+n_jac+1:offset] .= (n+1:n+m)

    pr_diag = view(V, 1:n)
    du_diag = view(V, n_jac+n_hess+n+1:n_jac+n_hess+n+m)

    hess = view(V, n+1:n+n_hess)
    jac = view(V, n_hess+n+1:n_hess+n+n_jac)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(m,n,jac_sparsity_I,jac_sparsity_J,jac)

    aug_com = MT(aug_raw)
    jac_com = MT(jac_raw)
    jac_prev = copy(jac_com)

    aug_csc_map = get_mapping(aug_com, aug_raw)
    jac_csc_map = get_mapping(jac_com, jac_raw)

    ind_aug_fixed = if isa(aug_com, SparseMatrixCSC)
        _get_fixed_variable_index(aug_com, ind_fixed)
    else
        zeros(Int, 0)
    end
    jac_scaling = ones(T, n_jac)

    last_grad = zeros(n)
    last_jacl = zeros(n)
    last_x = zeros(n)

    return LBFGSKKTSystem{T, MT}(
        lbfgs,
        hess, jac, pr_diag, du_diag,
        aug_raw, aug_com, aug_csc_map,
        jac_raw, jac_com, jac_prev, jac_csc_map,
        ind_ineq, ind_fixed, ind_aug_fixed, jac_scaling,
        last_grad, last_jacl, last_x,
    )
end

# Build KKT system directly from AbstractNLPModel
function LBFGSKKTSystem{T, MT}(nlp::AbstractNLPModel, ind_cons=get_index_constraints(nlp)) where {T, MT}
    n_slack = length(ind_cons.ind_ineq)
    # Deduce KKT size.
    n = get_nvar(nlp) + n_slack
    m = get_ncon(nlp)
    # Evaluate sparsity pattern
    jac_I = Vector{Int32}(undef, get_nnzj(nlp))
    jac_J = Vector{Int32}(undef, get_nnzj(nlp))
    jac_structure!(nlp, jac_I, jac_J)

    # Incorporate slack's sparsity pattern
    append!(jac_I, ind_cons.ind_ineq)
    append!(jac_J, get_nvar(nlp)+1:get_nvar(nlp)+n_slack)

    # LBFGS part
    lbfgs = CompactLBFGS(get_nvar(nlp))

    return LBFGSKKTSystem{T, MT}(
        n, m, ind_cons.ind_ineq, ind_cons.ind_fixed,
        lbfgs, jac_I, jac_J,
    )
end

is_reduced(::LBFGSKKTSystem) = true
num_variables(kkt::LBFGSKKTSystem) = length(kkt.pr_diag)

function compress_jacobian!(kkt::LBFGSKKTSystem{T, MT}) where {T, MT<:SparseMatrixCSC{T, Int32}}
    copyto!(nonzeros(kkt.jac_prev), nonzeros(kkt.jac_com)) # back-up previous values
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    kkt.jac .*= kkt.jacobian_scaling # scaling
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

