# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractMadNLPSolver{T} end

include("restoration.jl")

mutable struct MadNLPSolver{T,KKTSystem <: AbstractKKTSystem{T}} <: AbstractMadNLPSolver{T}
    nlp::AbstractNLPModel
    kkt::KKTSystem

    opt::MadNLPOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::Vector{T} # primal (after reformulation)
    y::Vector{T} # dual
    zl::Vector{T} # dual (after reformulation)
    zu::Vector{T} # dual (after reformulation)
    xl::Vector{T} # primal lower bound (after reformulation)
    xu::Vector{T} # primal upper bound (after reformulation)

    obj_val::T
    f::Vector{T}
    c::Vector{T}

    jacl::Vector{T}

    d::UnreducedKKTVector{T, Vector{T}}
    p::UnreducedKKTVector{T, Vector{T}}

    _w1::AbstractKKTVector{T, Vector{T}}
    _w2::AbstractKKTVector{T, Vector{T}}

    _w3::AbstractKKTVector{T, Vector{T}}
    _w4::AbstractKKTVector{T, Vector{T}}

    x_trial::Vector{T}
    c_trial::Vector{T}
    obj_val_trial::T

    x_slk::Vector{T}
    c_slk::SubVector{T}
    rhs::Vector{T}

    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    ind_llb::Vector{Int}
    ind_uub::Vector{Int}

    x_lr::SubVector{T}
    x_ur::SubVector{T}
    xl_r::SubVector{T}
    xu_r::SubVector{T}
    zl_r::SubVector{T}
    zu_r::SubVector{T}

    dx_lr::SubVector{T}
    dx_ur::SubVector{T}
    x_trial_lr::SubVector{T}
    x_trial_ur::SubVector{T}

    linear_solver::AbstractLinearSolver{T}
    iterator::AbstractIterator{T}

    obj_scale::Vector{T}
    con_scale::Vector{T}
    con_jac_scale::Vector{T}
    inf_pr::T
    inf_du::T
    inf_compl::T

    theta_min::T
    theta_max::T
    mu::T
    tau::T

    alpha::T
    alpha_z::T
    ftype::String

    del_w::T
    del_c::T
    del_w_last::T

    filter::Vector{Tuple{T,T}}

    RR::Union{Nothing,RobustRestorer{T}}
    status::Status
    output::Dict
end

function MadNLPSolver(nlp::AbstractNLPModel{T}; kwargs...) where T
    opt_ipm, opt_linear_solver, logger = load_options(; kwargs...)
    @assert is_supported(opt_ipm.linear_solver, T)

    VT = Vector{T}
    KKTSystem = if opt_ipm.kkt_system == SPARSE_KKT_SYSTEM
        MT = (input_type(opt_ipm.linear_solver) == :csc) ? SparseMatrixCSC{T, Int32} : Matrix{T}
        SparseKKTSystem{T, VT, MT}
    elseif opt_ipm.kkt_system == SPARSE_UNREDUCED_KKT_SYSTEM
        MT = (input_type(opt_ipm.linear_solver) == :csc) ? SparseMatrixCSC{T, Int32} : Matrix{T}
        SparseUnreducedKKTSystem{T, VT, MT}
    elseif opt_ipm.kkt_system == DENSE_KKT_SYSTEM
        MT = Matrix{T}
        DenseKKTSystem{T, VT, MT}
    elseif opt_ipm.kkt_system == DENSE_CONDENSED_KKT_SYSTEM
        MT = Matrix{T}
        DenseCondensedKKTSystem{T, VT, MT}
    end
    return MadNLPSolver{T,KKTSystem}(nlp, opt_ipm, opt_linear_solver; logger=logger)
end

# Inner constructor
function MadNLPSolver{T,KKTSystem}(
    nlp::AbstractNLPModel,
    opt::MadNLPOptions,
    opt_linear_solver::AbstractOptions;
    logger=MadNLPLogger(),
) where {T, KKTSystem<:AbstractKKTSystem{T}}
    cnt = MadNLPCounters(start_time=time())

    # generic options
    opt.disable_garbage_collector &&
        (GC.enable(false); @warn(logger,"Julia garbage collector is temporarily disabled"))
    set_blas_num_threads(opt.blas_num_threads; permanent=true)

    @trace(logger,"Initializing variables.")
    ind_cons = get_index_constraints(nlp; fixed_variable_treatment=opt.fixed_variable_treatment)
    ns = length(ind_cons.ind_ineq)
    n = get_nvar(nlp)+ns
    m = get_ncon(nlp)

    # Initialize KKT
    kkt = KKTSystem(nlp, ind_cons)

    xl = [get_lvar(nlp);view(get_lcon(nlp),ind_cons.ind_ineq)]
    xu = [get_uvar(nlp);view(get_ucon(nlp),ind_cons.ind_ineq)]
    x = [get_x0(nlp);zeros(T,ns)]
    y = copy(get_y0(nlp))
    zl= zeros(T,get_nvar(nlp)+ns)
    zu= zeros(T,get_nvar(nlp)+ns)

    f = zeros(T,n) # not sure why, but seems necessary to initialize to 0 when used with Plasmo interface
    c = zeros(T,m)

    n_jac = nnz_jacobian(kkt)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    x_trial=Vector{T}(undef,n)
    c_trial=Vector{T}(undef,m)

    x_slk= _madnlp_unsafe_wrap(x,ns, get_nvar(nlp)+1)
    c_slk= view(c,ind_cons.ind_ineq)
    rhs = (get_lcon(nlp).==get_ucon(nlp)).*get_lcon(nlp)

    x_lr = view(x, ind_cons.ind_lb)
    x_ur = view(x, ind_cons.ind_ub)
    xl_r = view(xl, ind_cons.ind_lb)
    xu_r = view(xu, ind_cons.ind_ub)
    zl_r = view(zl, ind_cons.ind_lb)
    zu_r = view(zu, ind_cons.ind_ub)
    x_trial_lr = view(x_trial, ind_cons.ind_lb)
    x_trial_ur = view(x_trial, ind_cons.ind_ub)

    if is_reduced(kkt)
        _w1 =  ReducedKKTVector{T,typeof(x)}(n, m)
        _w2 =  ReducedKKTVector{T,typeof(x)}(n, m)
        _w3 =  ReducedKKTVector{T,typeof(x)}(n, m)
        _w4 =  ReducedKKTVector{T,typeof(x)}(n, m)
    else
        _w1 = UnreducedKKTVector{T,typeof(x)}(n, m, nlb, nub)
        _w2 = UnreducedKKTVector{T,typeof(x)}(n, m, nlb, nub)
        _w3 = UnreducedKKTVector{T,typeof(x)}(n, m, nlb, nub)
        _w4 = UnreducedKKTVector{T,typeof(x)}(n, m, nlb, nub)
    end

    jacl = zeros(T,n) # spblas may throw an error if not initialized to zero

    d = UnreducedKKTVector{T,typeof(x)}(n, m, nlb, nub)
    dx_lr = view(d.xp, ind_cons.ind_lb) # TODO
    dx_ur = view(d.xp, ind_cons.ind_ub) # TODO

    p = UnreducedKKTVector{T,typeof(x)}(n, m, nlb, nub)

    obj_scale = T[1.0]
    con_scale = ones(T,m)
    con_jac_scale = ones(T,n_jac)

    @trace(logger,"Initializing linear solver.")
    cnt.linear_solver_time =
        @elapsed linear_solver = opt.linear_solver(get_kkt(kkt) ; opt=opt_linear_solver, logger=logger)

    n_kkt = size(kkt, 1)
    buffer_vec = similar(full(d), n_kkt)
    @trace(logger,"Initializing iterative solver.")
    iterator = opt.iterator(linear_solver, kkt, buffer_vec)

    @trace(logger,"Initializing fixed variable treatment scheme.")

    if opt.inertia_correction_method == INERTIA_AUTO
        opt.inertia_correction_method = is_inertia(linear_solver) ? INERTIA_BASED : INERTIA_FREE
    end


    return MadNLPSolver{T,KKTSystem}(nlp,kkt,opt,cnt,logger,
        n,m,nlb,nub,x,y,zl,zu,xl,xu,0.,f,c,
        jacl,
        d, p,
        _w1, _w2, _w3, _w4,
        x_trial,c_trial,0.,x_slk,c_slk,rhs,
        ind_cons.ind_ineq,ind_cons.ind_fixed,ind_cons.ind_llb,ind_cons.ind_uub,
        x_lr,x_ur,xl_r,xu_r,zl_r,zu_r,dx_lr,dx_ur,x_trial_lr,x_trial_ur,
        linear_solver,iterator,
        obj_scale,con_scale,con_jac_scale,
        0.,0.,0.,0.,0.,0.,0.,0.,0.," ",0.,0.,0.,
        Vector{T}[],nothing,INITIAL,Dict(),
    )
end

include("utils.jl")
include("kernels.jl")
include("callbacks.jl")
include("factorization.jl")
include("solver.jl")

