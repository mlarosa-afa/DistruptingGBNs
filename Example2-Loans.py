import numpy as np
from functions import *
from scipy.stats import invwishart, invgamma, multivariate_normal
from gb_SAA import gb_SAA
from wb_attack import whitebox_attack

MVG_Sigma_evidence = np.array([[4.185882e+09, -175182.27926,1410540839.1,563470612.6,6.280136e+09,   2.172400e+08,-8190.29377],
                               [-1.751823e+05,225.18981,    191876.1,    26066.5,    2.118087e+05,   8.711134e+03,10.52829],
                               [1.410541e+09, 191876.10111, 2882210596.6,414822294.7,5.245630e+09,   1.215825e+08,29660.39985],
                               [5.634706e+08, 26066.49798,  414822294.7, 706308549.2,2.062676e+09,   8.763860e+07,52309.89728],
                               [6.280136e+09, 211808.72361, 5245630110.1,2062675879.5,3.525107e+10,  5.879936e+08,26872.26282],
                               [2.172400e+08, 8711.13438,   121582519.9, 87638601.0, 5.879936e+08,   1.061313e+08,7505.96752],
                               [-8.190294e+03,10.52829,     29660.4,     52309.9,    2.687226e+04,   7.505968e+03,83.81531]])
MVG_mu_evidence = np.array(    [79411.67751,  19.30832,     51094.70827, 27390.36622,183760.49263,   16356.36090, 94.65052])

#Beta Cov matrix (True) PSD
beta_cov_matrix = generate_pos_def_matrix(8)

#Beta Mu (true)
mu_not_beta = np.random.normal(0,1,8)

mode = int(input("1) Whitebox Attack\n2) Graybox - Sample Average Approximation\nSelected Mode: "))
if mode == 1:
    U_1 = float(input("Enter U_1: "))
    U_2 = 1 - U_1
    print("Caluclated weight 2 as ", U_2)

    Sigma = MVG_Sigma_evidence
    mu = MVG_mu_evidence

    mu.append(mu_not_beta[0] + np.dot(mu_not_beta[1:], mu))

    new_row_sigma = np.array([])
    for j in range(len(mu_not_beta[1:])):
        new_row_sigma.append(np.dot(mu_not_beta[1:], Sigma[j,]))

    Sigma = np.append(Sigma, new_row_sigma[None].T, axis=1)
    new_row_sigma.append(0)
    Sigma = np.r_[Sigma, [new_row_sigma]]
    alpha = 4
    Sigma[8, 8] = invgamma.rvs(alpha, loc=0, scale=1) + mu_not_beta[1:] @ Sigma @ mu_not_beta[1:]

    ev_vars = [7]
    evidence = [6.5647]

    whitebox_attack(Sigma, mu, ev_vars, U_1, U_2)

elif mode == 2:
    U_1 = float(input("Enter U 1: "))
    U_2 = 1 - U_1
    numSamples = int(input("Enter number of Samples: "))

    Psi = MVG_Sigma_evidence
    nu = len(MVG_Sigma_evidence) + 2
    KAPPA = 4
    mu_not_evidence = MVG_mu_evidence

    #Cov/Mu for Evidence
    cov_samples_evidence = invwishart.rvs(nu, Psi, size=numSamples)

    mu_samples_evidence = []
    for cov_sample in cov_samples_evidence:
        mu_samples_evidence.append(np.random.multivariate_normal(mu_not_evidence, (1 / KAPPA) * cov_sample))

    #Sample LR Error
    alpha = 5
    errorVar_samples = invgamma.rvs(alpha, loc=0, scale=1, size=numSamples)

    #will need to change beta to static vars
    beta_samples = []
    for i in range(numSamples):
        beta_samples.append(np.random.multivariate_normal(mu_not_beta, beta_cov_matrix * errorVar_samples[i]))

    joint_cov = []
    joint_mu = []
    #loop through all samples and combind
    for i in range(len(cov_samples_evidence)):
        Sigma = cov_samples_evidence[i]
        mu = mu_samples_evidence[i]

        beta_sample = beta_samples[i]

        mu.append(beta_sample[0] + np.dot(beta_sample[1:], mu))

        new_row_sigma = np.array([])
        for j in range(len(beta_sample[1:])):
            new_row_sigma.append(np.dot(beta_sample[1:],Sigma[j,]))

        Sigma = np.append(Sigma, new_row_sigma[None].T, axis=1)
        new_row_sigma.append(0)
        Sigma = np.r_[Sigma, [new_row_sigma]]

        Sigma[8,8] = errorVar_samples[i] + beta_sample[1:] @ Sigma @ beta_sample[1:]

        joint_cov.append(Sigma)
        joint_mu.append(mu)

    ev_vars = [7]
    evidence = [6.5647]

    gb_SAA(cov_samples_evidence, mu_samples_evidence, ev_vars, evidence, U_1, U_2)