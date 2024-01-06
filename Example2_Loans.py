import numpy as np
from functions import *
from scipy.stats import invwishart, invgamma, multivariate_normal
from gb_SAA import gb_SAA
from wb_attack import whitebox_attack
import time

"""
MVG_Sigma_evidence = np.array([[4.185882e+09, -175182.27926,1410540839.1,563470612.6,6.280136e+09,   2.172400e+08,-8190.29377],
                               [-1.751823e+05,225.18981,    191876.1,    26066.5,    2.118087e+05,   8.711134e+03,10.52829],
                               [1.410541e+09, 191876.10111, 2882210596.6,414822294.7,5.245630e+09,   1.215825e+08,29660.39985],
                               [5.634706e+08, 26066.49798,  414822294.7, 706308549.2,2.062676e+09,   8.763860e+07,52309.89728],
                               [6.280136e+09, 211808.72361, 5245630110.1,2062675879.5,3.525107e+10,  5.879936e+08,26872.26282],
                               [2.172400e+08, 8711.13438,   121582519.9, 87638601.0, 5.879936e+08,   1.061313e+08,7505.96752],
                               [-8.190294e+03,10.52829,     29660.4,     52309.9,    2.687226e+04,   7.505968e+03,83.81531]])
MVG_mu_evidence = np.array(    [79411.67751,  19.30832,     51094.70827, 27390.36622,183760.49263,   16356.36090, 94.65052])
"""
MVG_Sigma_evidence = np.array([[4185.46278,	-175.16472,	1410.85166,	564.53087,	6279.87686,	217.373334,	-6.368214],
[-175.16472,	225.145394,	191.831712,	26.05891,	211.73991,	8.708476,	10.526218],
[1410.85166,	191.831712,	2881.78695,	416.01932,	5247.79571,	131.263828,	31.585219],
[564.530865,	26.058908,	416.019322,	706.81701,	2070.68296,	87.698081,	53.040351],
[6279.87686,	211.739907,	5247.79571,	2070.68296,	35241.1589,	588.131357,	41.619776],
[217.373334,	8.708476,	131.263828,	87.69808,	588.13136,	106.109665,	7.605169],
[-6.368214,	    10.526218,	31.585219,	53.04035,	41.61978,   7.605169,	83.798506]])
MVG_mu_evidence = np.array([79.41168,      19.30832,      51.09471,      27.39037,     183.76049,      16.35636,  94.65052])

#Beta Cov matrix (True) PSD
beta_cov_matrix = np.eye(8)

#Beta Mu (true)
mu_not_beta = [1.670487e+01, -2.041940e-06, 4.421601e-02, 1.133973e-05, -5.088275e-05, -3.370178e-06, 8.251364e-05, -5.166563e-02]

ev_vars = [0, 1, 2, 3, 4, 5, 6]
evidence = [90.000, 18.01, 38.767, 11.100, 70.795, 28.000, 92.9]
ev_bounds = np.stack(([x * 1.1 for x in evidence], [x * 0.9 for x in evidence]))
def gen_joint_distributions(cov_samples_evidence, mu_samples_evidence, beta_samples, errorVar_samples):
    joint_cov = []
    joint_mu = []
    # loop through all samples and combind
    for i in range(len(cov_samples_evidence)):
        Sigma = cov_samples_evidence[i]
        mu_evidence = mu_samples_evidence[i]
        beta_sample = beta_samples[i]
        errorVar = errorVar_samples[i]
        mu = np.append(mu_evidence, beta_sample[0] + np.dot(beta_sample[1:], mu_evidence))

        new_row_sigma = np.array([])
        for j in range(len(beta_sample[1:])):
            new_row_sigma = np.append(new_row_sigma, np.dot(beta_sample[1:], Sigma[j,]))
        #Calculate cov of the response
        cov_response = errorVar + beta_sample[1:] @ Sigma @ beta_sample[1:]
        #Add it as a row and then a col
        Sigma = np.append(Sigma, new_row_sigma[None].T, axis=1)
        new_row_sigma = np.append(new_row_sigma, 0)
        Sigma = np.r_[Sigma, [new_row_sigma]]

        Sigma[7, 7] = cov_response

        joint_cov.append(Sigma)
        joint_mu.append(mu)
    return joint_cov, joint_mu
def loan_wb(U_1, concavityFlag=False, timeFlag=False):
    np.random.seed(23)
    U_2 = 1 - U_1

    start_time = time.time()
    cov, mu = gen_joint_distributions([MVG_Sigma_evidence], [MVG_mu_evidence], [mu_not_beta], [21.66903])
    Sigma = cov[0]
    mu = mu[0]

    b_concave, b_convex, obj_val, solution, phi_opt1, phi_opt2 = whitebox_attack(Sigma, mu, ev_vars, evidence, U_1, U_2, ev_bounds=ev_bounds)
    end_time = time.time()
    print(U_1, U_2, solution["Z_DV_0"], solution["Z_DV_1"], solution["Z_DV_2"], solution["Z_DV_3"],
          solution["Z_DV_4"], solution["Z_DV_5"], solution["Z_DV_6"], obj_val, phi_opt1, phi_opt2, sep="\t")

    if timeFlag == True:
        print("Total time:\t", end_time - start_time)
    if concavityFlag == True:
        print("U_1-:", b_concave, "\tU_2+:", b_convex)

def loan_saa(U_1, numSamples, PsiMultiplier, mu_notMultiplier, KAPPA, nu, timeFlag=False):
    U_2 = 1 - U_1

    Psi = PsiMultiplier * MVG_Sigma_evidence
    mu_not_evidence = mu_notMultiplier * MVG_mu_evidence
    # phi_1opt = -26792.678576447775
    # phi_2opt = 1005.4561732538641
    phi_1opt, phi_2opt = saa_phi_opt_est(PsiMultiplier, mu_notMultiplier, KAPPA, nu)

    start_time = time.time()

    #Cov/Mu for Evidence
    cov_samples_evidence = invwishart.rvs(nu, Psi, size=numSamples)
    mu_samples_evidence = []
    for cov_sample in cov_samples_evidence:
        mu_samples_evidence.append(np.random.multivariate_normal(mu_not_evidence, (1 / KAPPA) * cov_sample))

    #Sample LR Error
    errorVar_samples = invgamma.rvs(4, loc=0, scale=2, size=numSamples)

    #will need to change beta to static vars
    beta_samples = []
    for i in range(numSamples):
        beta_cov_matrix = np.eye(8) * errorVar_samples[i]
        beta_samples.append(np.random.multivariate_normal(mu_not_beta, beta_cov_matrix))

    jcov, jmu = gen_joint_distributions(cov_samples_evidence, mu_samples_evidence, beta_samples, errorVar_samples)

    obj_val, solution, phi_1, phi_2 = gb_SAA(jcov, jmu, ev_vars, evidence, U_1, U_2, phi_1opt, phi_2opt, ev_bounds=ev_bounds)
    end_time = time.time()
    print(U_1, U_2, solution["Z_DV_0"], solution["Z_DV_1"], solution["Z_DV_2"], solution["Z_DV_3"],
          solution["Z_DV_4"], solution["Z_DV_5"], solution["Z_DV_6"], obj_val, phi_1, phi_2, sep="\t")

    if timeFlag == True:
        print("Total time:\t", end_time - start_time)

def saa_phi_opt_est(PsiMultiplier, mu_notMultiplier, KAPPA, nu, J=1000):
    Psi = PsiMultiplier * MVG_Sigma_evidence
    mu_not_evidence = mu_notMultiplier * MVG_mu_evidence

    # Cov/Mu for Evidence
    cov_samples_evidence = invwishart.rvs(nu, Psi, size=J)
    mu_samples_evidence = []
    for cov_sample in cov_samples_evidence:
        mu_samples_evidence.append(np.random.multivariate_normal(mu_not_evidence, (1 / KAPPA) * cov_sample))

    # Sample LR Error
    errorVar_samples = invgamma.rvs(4, loc=0, scale=2, size=J)

    # will need to change beta to static vars
    beta_samples = []
    for i in range(J):
        beta_cov_matrix = np.eye(8) * errorVar_samples[i]
        beta_samples.append(np.random.multivariate_normal(mu_not_beta, beta_cov_matrix))

    jcov, jmu = gen_joint_distributions(cov_samples_evidence, mu_samples_evidence, beta_samples, errorVar_samples)

    # Q, vT, c, K_prime, u_prime
    parameters = []
    # Convert Sample to normal form
    for i in range(len(jcov)):
        vals = vals_from_priors(jcov[i], jmu[i], ev_vars, evidence)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))  # Note uses canonical formto find Sigmazz^-1 and mu[Z}
    average_params = np.array(parameters, dtype=object)
    average_params = average_params.mean(axis=0)

    phi_opt1, phi_opt2 = solve_optimal_weights(average_params[0], average_params[1], average_params[3],
                                               average_params[4], len(ev_vars), ev_bounds=ev_bounds)

    return phi_opt1, phi_opt2
