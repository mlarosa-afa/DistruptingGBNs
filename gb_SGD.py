from scipy.stats import multivariate_normal
import numpy as np
from copy import copy
from functions import *
from sgd_algo import adaGrad, RMSProp, adam

def gb_SGD(solution, prev_solution, Psi, mu_not, ev_vars, evidence, method, W_1, W_2, constraint= None, risk_tolerance = .1, LEARN_RATE = .8, seed=12, nu = 100, KAPPA=4, error = .0001, epsilon = 0.00001):

    if constraint is None:
        constraint = np.stack(([x * (1 + risk_tolerance) for x in evidence], [x * (1 - risk_tolerance) for x in evidence]))

    Dmat = np.zeros((len(ev_vars), len(ev_vars)))
    Dvec = np.zeros(len(ev_vars))

    R = np.zeros((len(ev_vars), len(ev_vars)))
    t = 1
    m = 0
    np.random.seed(seed)
    while abs(np.linalg.norm(solution - prev_solution)) > error:
        sample_cov = invwishart.rvs(df=nu, scale=Psi)
        sample_Mu = multivariate_normal.rvs(mu_not, cov=(1/KAPPA)*sample_cov)

        # Q, vT, c, K_prime, u_prime
        vals = vals_from_priors(sample_cov, sample_Mu, ev_vars, evidence)

        Dmat = ((W_1 * vals.Q) - (W_2 * vals.K_prime))
        Dvec = W_1 * np.transpose(vals.vT) + 2 * W_2 * np.matmul(vals.K_prime, vals.u_prime)
        prev_solution = copy(solution)
        if method == 1:
            solution, R = adaGrad(lambda z: (Dmat + Dmat.transpose()) @ z + Dvec, constraint, solution,R,
                                  LEARN_RATE, epsilon=epsilon)
        elif method == 2:
            solution, R = RMSProp(lambda z: (Dmat + Dmat.transpose()) @ z + Dvec, constraint, solution,R,
                                  LEARN_RATE, R, epsilon=epsilon)
        elif method == 3:
            solution, R, m = adam(lambda z: (Dmat + Dmat.transpose()) @ z + Dvec, constraint, solution,R,
                                  LEARN_RATE, t=t, m=m, epsilon=epsilon)
            t = t + 1

    obj_val = solution @ Dmat @ solution + Dvec @ solution
    return obj_val, solution
