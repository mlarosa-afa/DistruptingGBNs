from np.scipy import multivariate_normal
import numpy as np
from functions import *
from sgd_algo import adaGrad, RMSProp, adam

def gb_SGD(MVG_Sigma, MVG_mu_true, ev_vars, evidence, method, U_1, U_2, constraint, LEARN_RATE = .8, seed=12, numDF = 100, error = .0001):

    #AM I USIG THE TRUE MU?

    Dmat = np.zeros((len(ev_vars), len(ev_vars)))
    Dvec = np.zeros(len(ev_vars))

    v = 0
    t = 1
    m = 0

    while abs(np.linalg.norm(solution - prev_solution)) > error:
        sample_cov = invwishart.rvs(df=numDF, scale=MVG_Sigma)
        sample_Mu = multivariate_normal.rvs(cov=sample_cov)

        # Q, vT, c, K_prime, u_prime
        vals = vals_from_priors(sample_cov, sample_Mu, ev_vars, evidence)

        Phi_opt1, Phi_opt2 = solve_optimal_weights(vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime, len(evidence))
        # Solve normalized problem
        W_1 = U_1 / Phi_opt1
        W_2 = U_2 / Phi_opt2

        Dmat = ((W_1 * vals.Q) - (W_2 * vals.K_prime))
        Dvec = W_1 * np.transpose(vals.vT) + 2 * W_2 * np.matmul(vals.K_prime, vals.u_prime)
        prev_solution = solution
        if method == 1:
            solution, v = adaGrad(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), constraint, solution,
                                  LEARN_RATE, v=v)
        elif method == 2:
            solution, v = RMSProp(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), constraint, solution,
                                  LEARN_RATE, v=v)
        elif method == 3:
            solution, v, m = adam(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), constraint, solution,
                                  LEARN_RATE, t=t, v=v, m=m)
            t = t + 1

    return solution
