

# Utility function 
function u(x,gamma) 
    if gamma==1
        return log.(x)
    else
        return x.^(1-gamma)./(1-gamma)
    end
end

#after the debt model is solved and simulated, the simulation is stored in "paths"
#get sequences of consumption and income from "paths"
function get_cons_path(paths::Vector, m)
    cons_path=map((j)->paths[j].c, collect(1:big_N))
    inc_path=map((j)->get_y_grid(m)[paths[j].y_ind], collect(1:big_N))
    return cons_path, inc_path
end;




#given two alternative sequences of consumption (labelled cons and inc), compute relative welfare of the first sequence over the second in consumption equivalents
#key inputs are discount rate and risk aversion parameters
function compute_value(cons, inc, beta, gamma)
    val_c_zero = 0.0
    val_y_zero = 0.0
    for (path_c, path_y) in zip(cons, inc) 
        val_c = 0.0
        val_y = 0.0 
        for (t, (c, y)) in enumerate(zip(path_c, path_y))
            val_c += beta^(t-1) * u(c, gamma)
            val_y += beta^(t-1) * u(y, gamma)
        end 
        val_c_zero += val_c 
        val_y_zero += val_y 
    end 
    val_zero = val_c_zero/length(cons)
    val_aut = val_y_zero/length(inc)
    return (val_zero/val_aut)^(1/(1-gamma))
end


#do welfare comparison for different discount factors
function do_beta_simulations!(beta_df, models, benchmark_parameters, shocks, paths, betaGrid)
    for j in eachindex(models)
        m=models[j]
        gamma0=benchmark_parameters.γ
        simulation!(paths, shocks, m; n = big_T, trim = 1000, trim_def = 20)
        cons_paths, inc_paths=get_cons_path(paths,m);
        cons_equiv = ThreadsX.map(beta->compute_value(cons_paths,inc_paths,beta,gamma0), betaGrid)
        insertcols!(beta_df, j=>cons_equiv)
    end
end 

#do welfare comparison for different risk aversions
function do_gamma_simulations!(gamma_df, models, benchmark_parameters, shocks, paths, gammaGrid)
    for j in eachindex(models)
        m=models[j]
        beta0= benchmark_parameters.pref.β #1/benchmark_parameters.R 
        simulation!(paths, shocks, m; n = big_T, trim = 1000, trim_def = 20)
        cons_paths, inc_paths=get_cons_path(paths,m);
        cons_equiv= ThreadsX.map(gamma->compute_value(cons_paths,inc_paths,beta0,gamma), gammaGrid)
        insertcols!(gamma_df, j=>cons_equiv)
    end
end 

#do welfare comparison of LoLR vs Runs for different discount factors
function norun_welfare_beta!(norun_beta_df, models,benchmark_parameters, shocks, paths, betaGrid)
    gamma0=benchmark_parameters.γ
    simulation!(paths, shocks, models.ckST; n = big_T, trim = 1000, trim_def = 20)
    cons_pathsCK=deepcopy(get_cons_path(paths,models.ckST)[1]);
    simulation!(paths, shocks, models.egST; n = big_T, trim = 1000, trim_def = 20)
    cons_pathsEG=deepcopy(get_cons_path(paths,models.egST)[1]);
    cons_equiv = ThreadsX.map(beta->compute_value(cons_pathsEG,cons_pathsCK,beta,gamma0), betaGrid)
    insertcols!(norun_beta_df, :Lambda=>cons_equiv)
end 

#do welfare comparison of LoLR vs Runs for different risk aversions
function norun_welfare_gamma!(norun_gamma_df, models,benchmark_parameters, shocks, paths, gammaGrid)
    beta0= benchmark_parameters.prefST.β
    simulation!(paths, shocks, models.egST; n = big_T, trim = 1000, trim_def = 20)
    cons_pathsEG=deepcopy(get_cons_path(paths,models.egST)[1]);
    simulation!(paths, shocks, models.ckST; n = big_T, trim = 1000, trim_def = 20)
    cons_pathsCK=deepcopy(get_cons_path(paths,models.ckST)[1]);
    cons_equiv = ThreadsX.map(gamma->compute_value(cons_pathsEG,cons_pathsCK,beta0,gamma), gammaGrid)
    insertcols!(norun_gamma_df, :Lambda=>cons_equiv)
end 

#for each discount factor in betaGrid, find risk aversion in gammaGrid that makes private agents indifferent to autarky
function do_frontier_simulations!(frontier, thresholds,m, shocks, paths, betaGrid,gammaGrid)
    simulation!(paths, shocks, m; n = big_T, trim = 1000, trim_def = 20)
    cons_paths, inc_paths=get_cons_path(paths,m);
    Threads.@threads for i in eachindex(betaGrid)
        for j in eachindex(gammaGrid) 
             frontier[i,j]=compute_value(cons_paths,inc_paths,betaGrid[i],gammaGrid[j])
        end
        thresholds[i]=findmin(x->abs(x-1),frontier[i,:])[2]
    end
end 

#for each discount factor in betaGrid, find risk aversion in gammaGrid that makes private agents indifferent to LoLR
function norun_do_frontier_simulations!(frontier, thresholds,models, shocks, paths, betaGrid,gammaGrid)
    simulation!(paths, shocks, models.ckST; n = big_T, trim = 1000, trim_def = 20)
    cons_pathsCK=deepcopy(get_cons_path(paths,models.ckST)[1]);
    simulation!(paths, shocks, models.egST; n = big_T, trim = 1000, trim_def = 20)
    cons_pathsEG=deepcopy(get_cons_path(paths,models.egST)[1]);
    Threads.@threads for i in eachindex(betaGrid)
        for j in eachindex(gammaGrid) 
             frontier[i,j]=compute_value(cons_pathsEG,cons_pathsCK,betaGrid[i],gammaGrid[j])
        end
        thresholds[i]=findmin(x->abs(x-1),frontier[i,:])[2]
    end
end 

