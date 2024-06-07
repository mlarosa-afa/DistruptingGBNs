from scipy.stats import multivariate_normal
import numpy as np
from copy import copy
from functions import *
from sgd_algo import adaGrad, RMSProp, adam

def gb_SGD(solution, prev_solution, Psi, mu_not, ev_vars, evidence, method, W_1, W_2, constraint= None, risk_tolerance = .1, LEARN_RATE = .8, seed=42, nu = 100, KAPPA=4, error = .01, epsilon = 0.0001):

    if constraint is None:
        constraint = np.stack(([x * (1 + risk_tolerance) for x in evidence], [x * (1 - risk_tolerance) for x in evidence]))

    Dmat = np.zeros((len(ev_vars), len(ev_vars)))
    Dvec = np.zeros(len(ev_vars))

    R = np.zeros((len(ev_vars), len(ev_vars)))
    t = 1
    m = 0
    np.random.seed(seed)
    while math.sqrt(np.linalg.norm(solution - prev_solution)) > error:
        sample_cov = invwishart.rvs(df=nu, scale=Psi)
        sample_Mu = multivariate_normal.rvs(mu_not, cov=(1/KAPPA)*sample_cov)

        # Q, vT, c, K_prime, u_prime
        vals = vals_from_priors(sample_cov, sample_Mu, ev_vars, evidence)

        Dmat = ((W_1 * vals.Q) - (W_2 * vals.K_prime))
        Dvec = W_1 * np.transpose(vals.vT) + 2 * W_2 * np.matmul(vals.K_prime, vals.u_prime)
        prev_solution = copy(solution)
        if method == 1:
            solution, R = adaGrad(lambda z: (Dmat + Dmat.transpose()) @ z + Dvec, constraint, solution, R,
                                  LEARN_RATE, epsilon=epsilon)
        elif method == 2:
            solution, R = RMSProp(lambda z: (Dmat + Dmat.transpose()) @ z + Dvec, constraint, solution,R,
                                  LEARN_RATE, R, epsilon=epsilon)
        elif method == 3:
            solution, R, m = adam(lambda z: (Dmat + Dmat.transpose()) @ z + Dvec, constraint, solution,R,
                                  LEARN_RATE, t=t, m=m, epsilon=epsilon)
            t = t + 1

    #Use SAA to estimate Obj Val of solution
    phi_1 = []
    phi_2 = []
    for i in range(10000):
        #Sample
        sample_cov = invwishart.rvs(df=nu, scale=Psi)
        sample_Mu = multivariate_normal.rvs(mu_not, cov=(1 / KAPPA) * sample_cov)

        #Change Forms
        vals = vals_from_priors(sample_cov, sample_Mu, ev_vars, evidence)

        #Calculate Phis
        phi_1_sample = solution @ (W_1*vals.Q) @ solution + solution @ (W_1 * np.transpose(vals.vT))
        phi_2_sample = solution @ (-1*W_2*vals.K_prime) @ solution + solution @ (2*W_2 * np.matmul(vals.K_prime, vals.u_prime))
        
        #record results
        phi_1.append(phi_1_sample)
        phi_2.append(phi_2_sample)

    #Get average
    phi_1 = np.average(phi_1)
    phi_2 = np.average(phi_2)


    return phi_1 + phi_2, solution, phi_1, phi_2
