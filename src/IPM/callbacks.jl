
function eval_f_wrapper(ips::InteriorPointSolver, x::Vector{Float64})
    nlp = ips.nlp
    cnt = ips.cnt
    @trace(ips.logger,"Evaluating objective.")
    cnt.eval_function_time += @elapsed obj_val = (get_minimize(nlp) ? 1. : -1.) * obj(nlp,view(x,1:get_nvar(nlp)))
    cnt.obj_cnt+=1
    cnt.obj_cnt==1 && (is_valid(obj_val) || throw(InvalidNumberException()))
    return obj_val*ips.obj_scale[]
end

function eval_grad_f_wrapper!(ips::InteriorPointSolver, f::Vector{Float64},x::Vector{Float64})
    nlp = ips.nlp
    cnt = ips.cnt
    @trace(ips.logger,"Evaluating objective gradient.")
    cnt.eval_function_time += @elapsed grad!(nlp,view(x,1:get_nvar(nlp)),view(f,1:get_nvar(nlp)))
    f.*=ips.obj_scale[] * (get_minimize(nlp) ? 1. : -1.)
    cnt.obj_grad_cnt+=1
    cnt.obj_grad_cnt==1 && (is_valid(f)  || throw(InvalidNumberException()))
    return f
end

function eval_cons_wrapper!(ips::InteriorPointSolver, c::Vector{Float64},x::Vector{Float64})
    nlp = ips.nlp
    cnt = ips.cnt
    @trace(ips.logger, "Evaluating constraints.")
    cnt.eval_function_time += @elapsed cons!(nlp,view(x,1:get_nvar(nlp)),view(c,1:get_ncon(nlp)))
    view(c,ips.ind_ineq).-=view(x,get_nvar(nlp)+1:ips.n)
    c.-=ips.rhs
    c.*=ips.con_scale
    cnt.con_cnt+=1
    cnt.con_cnt==2 && (is_valid(c) || throw(InvalidNumberException()))
    return c
end

function eval_jac_wrapper!(ipp::InteriorPointSolver, kkt::AbstractKKTSystem, x::Vector{Float64})
    nlp = ipp.nlp
    cnt = ipp.cnt
    ns = length(ipp.ind_ineq)
    @trace(ipp.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_coord!(nlp,view(x,1:get_nvar(nlp)),view(jac,1:get_nnzj(nlp)))
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException()))
    @trace(ipp.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(ipp::InteriorPointSolver, kkt::AbstractKKTSystem, x::Vector{Float64},l::Vector{Float64};is_resto=false)
    nlp = ipp.nlp
    cnt = ipp.cnt
    @trace(ipp.logger,"Evaluating Lagrangian Hessian.")
    dual(ipp._w1) .= l.*ipp.con_scale
    hess = get_hessian(kkt)
    cnt.eval_function_time += @elapsed hess_coord!(
        nlp, view(x,1:get_nvar(nlp)), dual(ipp._w1), view(hess, 1:get_nnzh(nlp));
        obj_weight = (get_minimize(nlp) ? 1. : -1.) * (is_resto ? 0.0 : ipp.obj_scale[]))
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return hess
end

function eval_jac_wrapper!(ipp::InteriorPointSolver, kkt::AbstractDenseKKTSystem, x::Vector{Float64})
    nlp = ipp.nlp
    cnt = ipp.cnt
    ns = length(ipp.ind_ineq)
    @trace(ipp.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_dense!(nlp,view(x,1:get_nvar(nlp)),jac)
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException()))
    @trace(ipp.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(ipp::InteriorPointSolver, kkt::AbstractDenseKKTSystem, x::Vector{Float64},l::Vector{Float64};is_resto=false)
    nlp = ipp.nlp
    cnt = ipp.cnt
    @trace(ipp.logger,"Evaluating Lagrangian Hessian.")
    dual(ipp._w1) .= l.*ipp.con_scale
    hess = get_hessian(kkt)
    cnt.eval_function_time += @elapsed hess_dense!(
        nlp, view(x,1:get_nvar(nlp)), dual(ipp._w1), hess;
        obj_weight = (get_minimize(nlp) ? 1. : -1.) * (is_resto ? 0.0 : ipp.obj_scale[]))
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return hess
end

function eval_lag_hess_wrapper!(
    ips::InteriorPointSolver,
    kkt::LBFGSKKTSystem,
    x::Vector{Float64},
    l::Vector{Float64};
    is_resto=false,
)
    nlp = ips.nlp
    cnt = ips.cnt
    B = kkt.lbfgs
    n, p = size(B)
    @trace(ips.logger,"Update LBFGS matrices.")

    # Make sure we have evaluated the gradient before.
    if ips.cnt.obj_grad_cnt > 0
        ∇Lk = ips.f
        mul!(∇Lk, kkt.jac_com', l, 1.0, 1.0)

        ∇Lp = kkt.last_grad
        mul!(∇Lp, kkt.jac_prev', l, 1.0, 1.0)

        # Update LBFGS information.
        sk = (ips.x .- kkt.last_x)[1:n]
        yk = (∇Lk .- ∇Lp)[1:n]
        update_values!(B, sk, yk)
    end

    copyto!(kkt.last_x, ips.x)
    copyto!(kkt.last_grad, ips.f)

    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    # cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return get_hessian(kkt)
end

