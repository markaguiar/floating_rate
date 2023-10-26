# stuff for IMF conference

using Revise
using LTBonds
using ThreadsX 
using Random 
using PrettyTables
using Plots
using LaTeXStrings
using Infiltrator
using DataFrames


#get paths of consumption and income 
function get_cons_path(paths::Vector, shocks_vec::Vector, m)
    simulation!(paths, shocks_vec, m; n = big_T, trim = 1000, trim_def = 20)
    cons_path=map((j)->paths[j].c, collect(1:big_N))
    inc_path=map((j)->paths[j].y, collect(1:big_N))
    return cons_path, inc_path
end;



function u(x,gamma) 
    if gamma==1
        return log.(x)
    else
        return x.^(1-gamma)./(1-gamma)
    end
end



function compute_value(cons, inc, beta, gamma)
    out= map(x->u(x,gamma),cons)
    val_zero=0.0
    for path in out
        val=0.0
        for (t,flow) in enumerate(path)
            val += beta^(t-1)*flow
        end 
        val_zero+=val
    end

    out= map(x->u(x,gamma),inc)
    val_aut=0.0
    for path in out
        val=0.0
        for (t,flow) in enumerate(path)
            val += beta^(t-1)*flow
        end 
        val_aut+=val
    end

    val_zero=val_zero/length(cons)
    val_aut=val_aut/length(inc)


    return (val_zero/val_aut)^(1/(1-gamma))
end
   

benchmark_parameters =  let
    R = 1.01
    β = 0.9540232420
    γ=2
    pref = Preferences(β = β, u = make_CRRA(ra = γ))
    y = discretize(YProcess(n = 50, ρ = 0.948503, std = 0.027092, μ = 0.0, span = 3.0, tails = false))
    m = MTruncatedNormal(; std = 0.003, span = 2.0, quadN = 100)
    penalty = DefCosts(pen1 = -0.1881927550, pen2 = 0.2455843389, quadratic = true, reentry = 0.0385)
    η=0.1

    (R = R, pref = pref, y = y, m = m, penalty = penalty, η = η,γ=γ)
end;

models = let
    R, pref, y, m, penalty, η = benchmark_parameters

    #bondFR = FloatingRateBond(;n = 350, min = 0.0, max = 1.5, λ = 0.05, κbar = 0.06) 
    #bondFRlowκ = FloatingRateBond(;n = 350, min = 0.0, max = 1.5, λ = 0.05, κbar = 0.015) 
   # bondLT = Bond(n = 350, min = 0.0, max = 1.5, κ = R - 1, λ = 0.05) 
   bondLT = Bond(n = 350, min = 0.0, max = 1.5, κ = 0.03, λ = 0.05)  
    bondST = Bond(n = 350, min = 0.0, max = 1.5, κ = R - 1, λ = 1.0)  
    

    egLT = generate_workspace(
        LTBondModel(
            y = y,
            m = m, 
            preferences = pref, 
            bond = bondLT, 
            def_costs = penalty, 
            R = R,
        )
    )

    egST = generate_workspace(
        LTBondModel(
            y = y,
            m = m, 
            preferences = pref, 
            bond = bondST, 
            def_costs = penalty, 
            R = R,
        )
    )

    ckST = generate_workspace(
        CKLTBondModel(
            y = y,
            m = m, 
            preferences = pref, 
            bond = bondST, 
            def_costs = penalty, 
            R = R,
            η = η
        )
    )


    (; egLT, egST, ckST) 
end;

# compute models
@time for m ∈ models
    @time solve!(m; print_every = 200, max_iters = 5_000, err = 1e-7)
end


#simulations
big_T = 20_000 
big_N = 1_000

#generate shock path 
rng = Random.seed!(1234)
shocks, paths = create_shocks_paths(models.ckST, big_T, big_N; rng) # make sure to use a ck model to draw the sunspots
betaGrid=LinRange(0.95,0.999,50)
beta_df=DataFrame("Beta"=>betaGrid)

println("Simulations")

for j in eachindex(models)
    m=models[j]
    beta0=benchmark_parameters.pref.β
    gamma0=benchmark_parameters.γ
    cons_paths, inc_paths=get_cons_path(paths,shocks,m);
    cons_equiv= ThreadsX.map(beta->compute_value(cons_paths,inc_paths,beta,gamma0), betaGrid)
    insertcols!(beta_df, j=>cons_equiv)
end

gammaGrid=LinRange(0.5,10,50)
gamma_df=DataFrame("Gamma"=>gammaGrid)

for j in eachindex(models)
    m=models[j]
    beta0= 0.98 #benchmark_parameters.pref.β #1/benchmark_parameters.R 
    gamma0=benchmark_parameters.γ
    cons_paths, inc_paths=get_cons_path(paths,shocks,m);
    cons_equiv= ThreadsX.map(gamma->compute_value(cons_paths,inc_paths,beta0,gamma), gammaGrid)
    insertcols!(gamma_df, j=>cons_equiv)
end



# Some plots
rho= -4*log.(betaGrid)
insertcols!(beta_df,2,:rho=>rho)

rstar = 4 * log(benchmark_parameters.R)
rho0 = - 4 * log(benchmark_parameters.pref.β)

plot(beta_df.rho,100*(beta_df.egLT.-1),color=:black,lw=3,legend=false)
plot!(beta_df.rho,100*(beta_df.egST.-1),ls=:dash,color=:black,lw=3)
plot!(beta_df.rho,100*(beta_df.ckST.-1),ls=:dashdot,color=:green,lw=3)
xlims!(0.0,0.25)
ylims!(-2,4)
yticks!([-2, -1, 0, 1, 2, 3, 4],["-2%","-1%","0","1%","2%","3%","4%"])
xlabel!("Household Discount Rate: " * L"\rho")
hline!([0],color=:black, lw=1)
vline!([rstar,rho0],color=:LightGray,lw=3)


plot(gamma_df.Gamma,100*(gamma_df.egLT.-1),color=:black,lw=3,legend=false)
plot!(gamma_df.Gamma,100*(gamma_df.egST.-1),ls=:dash,color=:black,lw=3)
plot!(gamma_df.Gamma,100*(gamma_df.ckST.-1),ls=:dashdot,color=:green,lw=3)
xlims!(0.4,10)
ylims!(-2,2)
yticks!([-2, -1, 0, 1, 2],["-2%","-1%","0","1%","2%"])
xlabel!("Household Risk Aversion: " * L"\gamma")
hline!([0],color=:black, lw=1)
vline!([benchmark_parameters.γ],color=:LightGray,lw=3)


