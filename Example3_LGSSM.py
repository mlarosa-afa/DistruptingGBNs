import numpy as np
from scipy.stats import invgamma, multivariate_normal
from functions import solve_optimal_weights, vals_from_priors
from gb_SAA import gb_SAA
from wb_attack import whitebox_attack, evaluate_objective, whitebox_evaluation
from functions import *
import time
from tabulate import tabulate
from baseline_analysis import random_attack

np.random.seed(100)
evidence = [0.1, 0.2, 1.9, 1.1, 3.8, 2.3, 6.1, 3.1, 7.9, 4.2, 10.1, 5.1, 12.2, 5.9, 13.9, 7.1, 15.9, 8.2,
                18.1, 9.4, 19.9, 10.2]  # multiples of 6 starting with 4 and 5 (zero indexed)
T = int(len(evidence) / 2)
ev_vars = []  # multiples of 6 starting with 4 and 5 (zero indexed)
i = 4
while i < T * 6:
    ev_vars.append(i)
    ev_vars.append(i + 1)
    i = i + 6

def LGSSM_Generate(T, epsilon_var=None, delta_var=None, mu_state_init=None, var_state_init=None, seed=23):
    np.random.seed(seed)
    if epsilon_var is None:
        epsilon_var = []
        for i in [.1, .1, .15811, .15811]:
            epsilon_var.append(invgamma.rvs(2, loc=0, scale=i, size=1)[0])
    if delta_var is None:
        delta_var = []
        for i in [0.2, 0.2]:
            delta_var.append(invgamma.rvs(2, loc=0, scale=i, size=1)[0])
    if mu_state_init is None:
        mu_state_init = multivariate_normal.rvs(mean=[0, 0, 2, 1], cov=np.eye(4), size=1, random_state=None)
    if var_state_init is None:
        var_state_init = []
        for i in [.1, .1, .25, .25]:
            var_state_init.append(invgamma.rvs(2, loc=0, scale=i, size=1)[0])
    Delta_t = 1

    # Build matrix of beta coefficients
    beta = np.zeros((6 * (T + 1), 6 * (T + 1)))  # Initialize to zero matrix
    beta[4][0] = 1
    beta[5][1] = 1
    for t in range(1, T + 1):
        # Coef for state variables
        beta[6 * t][(6 * (t - 1)):(6 * (t - 1) + 6)] = 1, 0, Delta_t, 0, 0, 0
        beta[6 * t + 1][(6 * (t - 1)):(6 * (t - 1) + 6)] = 0, 1, 0, Delta_t, 0, 0
        beta[6 * t + 2][(6 * (t - 1)):(6 * (t - 1) + 6)] = 0, 0, 1, 0, 0, 0
        beta[6 * t + 3][(6 * (t - 1)):(6 * (t - 1) + 6)] = 0, 0, 0, 1, 0, 0
        # Coef for emissions(obs)
        beta[6 * t + 4, 6 * t] = 1
        beta[6 * t + 5, 6 * t + 1] = 1

    mu = np.zeros((6 * (T + 1)))  # Initialize

    # Find mu of joint PDF -- loop thru Koller's recursion
    for t in range(T + 1):
        if t == 0:
            for i in range(5):  # Update all state/obs means
                if i < 4:
                    mu[i - 1] = mu_state_init[i - 1]  # Position and velocities
                else:
                    mu[i - 1] = mu[i - 4 - 1]  # Sensor readings

        else:  # when t>0
            for i in range(6):  # Update all state/obs means
                if i <= 1:
                    mu[6 * t + i] = mu[6 * (t - 1) + i] + Delta_t * mu[6 * (t - 1) + (i + 2)]  # Position
                elif i == 2 or i == 3:
                    mu[6 * t + i] = mu[6 * (t - 1) + i]  # Velocities
                else:
                    mu[6 * t + i] = mu[6 * t + (i - 4)]  # Sensor readings

    Sigma = np.zeros((6 * (T + 1), 6 * (T + 1)))  # Initialize to zero matrix

    for i in range((6 * (T + 1))):
        for j in range(i + 1):
            # Check if i=j and t=0
            if i == j and i <= 5:
                if i <= 3:
                    Sigma[i, j] = var_state_init[i]
                else:
                    Sigma[i, j] = delta_var[i - 4] + np.transpose(beta[i, :(j - 1)]) @ Sigma[:(j - 1), :(j - 1)] @ beta[
                                                                                                                   i, :(
                            j - 1)]

            # Check if i=j and t>0
            if i == j and i >= 6:
                if i % 6 <= 3 and i % 6 >= 0:
                    Sigma[i, j] = epsilon_var[i % 6] + np.transpose(beta[i, :(j - 1)]) @ Sigma[:(j - 1),
                                                                                         :(j - 1)] @ beta[i, :(j - 1)]
                else:
                    if i % 6 == 4:
                        Sigma[i, j] = delta_var[1] + np.transpose(beta[i, :(j - 1)]) @ Sigma[:(j - 1), :(j - 1)] @ beta[
                                                                                                                   i, :(
                                j - 1)]
                    else:
                        Sigma[i, j] = delta_var[0] + np.transpose(beta[i, :(j - 1)]) @ Sigma[:(j - 1), :(j - 1)] @ beta[
                                                                                                                   i, :(
                                j - 1)]

            # Check if i!=j and i>5 (i.e., we are at the first y10 value - at least)
            if i != j and i >= 3:
                Sigma[i, j] = np.transpose(Sigma[j, :(i - 1)]) @ beta[i, :(i - 1)]
                Sigma[j, i] = Sigma[i, j]

    return Sigma, mu

def lgssm_baseline(U_1, risk_tolerance, concavityFlag=False, timeFlag=False, verbose=True):
    start_time = time.time()
    MVG_Sigma, MVG_mu = LGSSM_Generate(T, epsilon_var=[.1, .1, .15811, .15811], delta_var=[0.2, 0.2],
                                       mu_state_init=[0, 0, 2, 1], var_state_init=[.1, .1, .25, .25])
    obj_value, phi_1, phi_2 = evaluate_objective(MVG_Sigma, MVG_mu, ev_vars, evidence, evidence, U_1, risk_tolerance=risk_tolerance)
    end_time = time.time()
    
    if verbose:
        print(f"Solution: LGSSM Baseline Analysis\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}")
        print("\nEvidence Analytics")
        print(tabulate([["Objective Value", obj_value]]))
        print(f"\nEstimates of Unobserved Means")
        point_estimates, sd_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, evidence)
        print(tabulate([["Y_"+str(i), point_estimates[i], sd_estimates[i]] for i in range(len(point_estimates))], headers=['Variable', 'Point Estimates', "Standard Deviation"]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, risk_tolerance=risk_tolerance)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
            
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))
    
    return obj_value, phi_1, phi_2

def lgssm_random(U_1, risk_tolerance, concavityFlag=False, timeFlag=False, verbose=True, seed=2023):
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    MVG_Sigma, MVG_mu = LGSSM_Generate(T, epsilon_var=[.1, .1, .15811, .15811], delta_var=[0.2, 0.2],
                                       mu_state_init=[0, 0, 2, 1], var_state_init=[.1, .1, .25, .25])
    obj_val, solution, phi_1, phi_2 = random_attack(MVG_Sigma, MVG_mu, ev_vars, evidence, U_1, risk_tolerance=risk_tolerance)
    end_time = time.time()

    #print(f"Objective Value with random attack: {obj_value} \n Utilized random evidence:{proposed_evidence}")
    if verbose:
        print(f"Solution: LGSSM Random Analysis\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}")
        print("\nPoisioned Attack")
        print(tabulate([["Z_"+str(i), round(solution[i],3)] for i in range(len(solution))], headers=['Evidence', 'Posioned Value']))
        print("\nAttack Analytics")
        print(tabulate([["Objective Value", obj_val],["Phi_1",phi_1],["Phi_2",phi_2]]))
        print(f"\nKL Divergence from True Evidence")
        print(tabulate([["KL", KL_divergence(MVG_Sigma, MVG_mu, ev_vars, evidence, solution)]]))
        print(f"\nEstimates of Unobserved Means")
        point_estimates, sd_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, evidence)
        print(tabulate([["Y_"+str(i), point_estimates[i], sd_estimates[i]] for i in range(len(point_estimates))], headers=['Variable', 'Point Estimates', "Standard Deviation"]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, risk_tolerance=risk_tolerance)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
            
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))
    
    return obj_val, solution, phi_1, phi_2

def lgssm_wb(U_1, risk_tolerance, concavityFlag=False, timeFlag=False, verbose=True):
    ev_bounds = np.stack(([x * (1+risk_tolerance) for x in evidence], [x * (1-risk_tolerance) for x in evidence]))

    # build LG-SSM based off of evidence
    start_time = time.time()
    T = int(len(evidence) / 2)
    MVG_Sigma, MVG_mu = LGSSM_Generate(T, epsilon_var=[.1, .1, .15811, .15811], delta_var=[0.2, 0.2],
                                       mu_state_init=[0, 0, 2, 1], var_state_init=[.1, .1, .25, .25])

    obj_val, solution, phi_1, phi_2 = whitebox_attack(MVG_Sigma, MVG_mu, ev_vars, evidence, U_1, ev_bounds=ev_bounds)
    end_time = time.time()

    if verbose:
        print(f"Solution: LGSSM Whitebox Attack\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}")
        print("\nPoisioned Attack")
        print(tabulate([["Z_"+str(i), round(solution[i],3)] for i in range(len(solution))], headers=['Evidence', 'Posioned Value']))
        print("\nAttack Analytics")
        print(tabulate([["Objective Value", obj_val],["Phi_1",phi_1],["Phi_2",phi_2]]))
        print(f"\nKL Divergence from True Evidence")
        print(tabulate([["KL", KL_divergence(MVG_Sigma, MVG_mu, ev_vars, evidence, solution)]]))
        print(f"\nPoint Estimates of Unobserved Means")
        point_estimates, sd_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, solution)
        print(tabulate([["Y_"+str(i), point_estimates[i], sd_estimates[i]] for i in range(len(point_estimates))], headers=['Variable', 'Point Estimates', "Standard Deviation"]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds=ev_bounds)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
        
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))

    return obj_val, solution, phi_1, phi_2

phi_1opt, phi_2opt = 26431474.87572283, 29.634639300644544
def lgssm_saa(U_1, numSamples, risk_tolerance,  concavityFlag=False, timeFlag=False, verbose=True):
    ev_bounds = np.stack(([x * (1+risk_tolerance) for x in evidence], [x * (1-risk_tolerance) for x in evidence]))
    #phi_1opt, phi_2opt = saa_phi_opt_est(risk_tolerance=risk_tolerance)

    start_time = time.time()
    cov_samples = []
    mu_samples = []
    for i in range(numSamples):
        cov_sample, mu_sample = LGSSM_Generate(T)
        cov_samples.append(cov_sample)
        mu_samples.append(mu_sample)

    obj_val, solution, phi_1, phi_2 = gb_SAA(cov_samples, mu_samples, ev_vars, evidence, U_1/ abs(phi_1opt), (1-U_1)/abs(phi_2opt), ev_bounds=ev_bounds)
    end_time = time.time()

    #Point Estimates for samples
    MVG_Sigma = np.array(cov_samples).mean(axis=0)
    MVG_mu = np.array(mu_samples).mean(axis=0)

    if verbose:
        print(f"Solution: LGSSM Grey Box (SAA) Attack\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}, r_rol = {risk_tolerance}")
        print("\nPoisioned Attack")
        print(tabulate([["Z_"+str(i), round(solution[i],3)] for i in range(len(solution))], headers=['Evidence', 'Posioned Value']))
        print("\nAttack Analytics")
        print(tabulate([["Objective Value", obj_val],["Phi_1",phi_1],["Phi_2",phi_2]]))

        ##########################################
        ###Evaluating GB Solution in WB Setting###
        ##########################################
        v_true = vals_from_priors(MVG_Sigma, MVG_mu, ev_vars, evidence)
        #Calculate Normalized Weights
        phi_opt1, phi_opt2 = solve_optimal_weights(v_true.Q, v_true.vT, v_true.K_prime, v_true.u_prime, len(ev_vars), ev_bounds=ev_bounds)
        W_1_true = U_1 / abs(phi_opt1)
        W_2_true = (1-U_1) / abs(phi_opt2)

        wb_obj, wb_phi_1, wb_phi_2 = whitebox_evaluation(v_true, W_1_true, W_2_true, solution)
        print(f"\nWhitebox Evaluation of Solution")
        print(tabulate([["Objective Value", wb_obj],["Phi_1", wb_phi_1],["Phi_2", wb_phi_2]]))

        print(f"\nKL Divergence from True Evidence")
        print(tabulate([["KL", KL_divergence(MVG_Sigma, MVG_mu, ev_vars, evidence, solution)]]))
        print(f"\nEstimates of Unobserved Means")
        point_estimates, sd_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, solution)
        print(tabulate([["Y_"+str(i), point_estimates[i], sd_estimates[i]] for i in range(len(point_estimates))], headers=['Variable', 'Point Estimates', "Standard Deviation"]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds=ev_bounds)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
        
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))

    return obj_val, solution, phi_1, phi_2
    




def saa_phi_opt_est(J=1000, risk_tolerance = 0.1):
    cov_samples = []
    mu_samples = []
    for i in range(J):
        cov, mu = LGSSM_Generate(11)
        cov_samples.append(cov)
        mu_samples.append(mu)

    # Q, vT, c, K_prime, u_prime
    parameters = []
    # Convert Sample to normal form
    for i in range(len(cov_samples)):
        vals = vals_from_priors(cov_samples[i], mu_samples[i], ev_vars, evidence)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))  # Note uses canonical formto find Sigmazz^-1 and mu[Z}
    average_params = np.array(parameters, dtype=object)
    average_params = average_params.mean(axis=0)

    ev_bounds = np.stack(([x * (1 + risk_tolerance) for x in evidence], [x * (1 - risk_tolerance) for x in evidence]))

    phi_opt1, phi_opt2 = solve_optimal_weights(average_params[0], average_params[1], average_params[3],
                                               average_params[4], len(ev_vars), ev_bounds=ev_bounds)

    return phi_opt1, phi_opt2

