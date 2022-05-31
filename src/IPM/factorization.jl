
function factorize_wrapper!(ips::InteriorPointSolver)
    @trace(ips.logger,"Factorization started.")
    build_kkt!(ips.kkt)
    ips.cnt.linear_solver_time += @elapsed factorize!(ips.linear_solver)
end

function solve_refine_wrapper!(
    ips::InteriorPointSolver,
    x::AbstractKKTVector,
    b::AbstractKKTVector,
)
    cnt = ips.cnt
    @trace(ips.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(values(b), ips.ind_fixed)

    cnt.linear_solver_time += @elapsed begin
        result = solve_refine!(x, ips.iterator, b)
    end

    if result == :Solved
        solve_status =  true
    else
        if improve!(ips.linear_solver)
            cnt.linear_solver_time += @elapsed begin
                factorize!(ips.linear_solver)
                ret = solve_refine!(x, ips.iterator, b)
                solve_status = (ret == :Solved)
            end
        else
            solve_status = false
        end
    end
    fixed_variable_treatment_vec!(values(x), ips.ind_fixed)
    return solve_status
end

function solve_refine_wrapper!(
    ips::InteriorPointSolver{<:DenseCondensedKKTSystem},
    x::AbstractKKTVector,
    b::AbstractKKTVector,
)
    cnt = ips.cnt
    @trace(ips.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(values(b), ips.ind_fixed)

    kkt = ips.kkt

    n = num_variables(kkt)
    n_eq, ns = kkt.n_eq, kkt.n_ineq
    n_condensed = n + n_eq

    # load buffers
    b_c = view(values(ips._w1), 1:n_condensed)
    x_c = view(values(ips._w2), 1:n_condensed)
    jv_x = view(values(ips._w3), 1:ns) # for jprod
    jv_t = primal(ips._w4)             # for jtprod
    v_c = dual(ips._w4)

    Σs = get_slack_regularization(kkt)
    α = get_scaling_inequalities(kkt)

    # Decompose right hand side
    bx = view(values(b), 1:n)
    bs = view(values(b), n+1:n+ns)
    by = view(values(b), kkt.ind_eq_shifted)
    bz = view(values(b), kkt.ind_ineq_shifted)

    # Decompose results
    xx = view(values(x), 1:n)
    xs = view(values(x), n+1:n+ns)
    xy = view(values(x), kkt.ind_eq_shifted)
    xz = view(values(x), kkt.ind_ineq_shifted)

    v_c .= 0.0
    v_c[kkt.ind_ineq] .= (Σs .* bz .+ α .* bs) ./ α.^2
    jtprod!(jv_t, kkt, v_c)
    # init right-hand-side
    b_c[1:n] .= bx .+ jv_t[1:n]
    b_c[1+n:n+n_eq] .= by

    cnt.linear_solver_time += @elapsed (result = solve_refine!(x_c, ips.iterator, b_c))
    solve_status = (result == :Solved)

    # Expand solution
    xx .= x_c[1:n]
    xy .= x_c[1+n:end]
    jprod_ineq!(jv_x, kkt, xx)
    xz .= sqrt.(Σs) ./ α .* jv_x .- Σs .* bz ./ α.^2 .- bs ./ α
    xs .= (bs .+ α .* xz) ./ Σs

    fixed_variable_treatment_vec!(values(x), ips.ind_fixed)
    return solve_status
end

function solve_refine_wrapper!(
    ips::InteriorPointSolver{<:LBFGSKKTSystem},
    x::AbstractKKTVector,
    b::AbstractKKTVector,
)
    kkt = ips.kkt
    lbfgs = kkt.lbfgs
    Tk = lbfgs.Tk
    n, p = size(lbfgs)

    cnt = ips.cnt
    @trace(ips.logger,"Iterative solution started.")
    fixed_variable_treatment_vec!(values(b), ips.ind_fixed)

    # Solve linear system without low-rank parts
    cnt.linear_solver_time += @elapsed begin
        result = solve_refine!(x, ips.iterator, b)
    end

    if p > 0
        # Solve C⁻¹ U
        copyto!(lbfgs.Utilde, lbfgs.U)
        multi_solve!(ips.linear_solver, lbfgs.Utilde)

        # Sherman-Morrison-Woodbury formula
        # T = (I + Uᵀ C⁻¹ U)
        fill!(Tk, 0.0)
        Tk[LinearAlgebra.diagind(Tk)] .= 1.0
        mul!(Tk, lbfgs.U', lbfgs.U, 1.0, 1.0)

        # Factorize
        J1 = LinearAlgebra.cholesky(Tk)

        # add low-rank correction to x
        b1 = view(values(b), 1:n)
        x1 = view(values(x), 1:n)
        x2 = zeros(2*p)
        mul!(x2, lbfgs.Utilde', b1)
        ldiv!(J1, x2)
        mul!(x1, lbfgs.Utilde, x2)
    end

    fixed_variable_treatment_vec!(values(x), ips.ind_fixed)
    solve_status = (result == :Solved)
    return solve_status
end

