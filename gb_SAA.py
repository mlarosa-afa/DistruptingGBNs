from cplex import Model
import numpy as np
from scipy.stats import invwishart, multivariate_normal
from functions import *

def gb_SAA(cov_samples, mu_samples, ev_cols, evidence, U_1, U_2, seed=12):
    unob_cols = list(set(list(range(len(cov_samples[1][0])))) - set(ev_cols))

    # Q, vT, c, K_prime, u_prime
    parameters = []

    for i in range(len(cov_samples)):

        vals = vals_from_priors(cov_samples[i], mu_samples[i], ev_cols, unob_cols, evidence)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))

    Dmat = np.zeros((len(ev_cols), len(ev_cols)))
    Dvec = np.zeros(len(ev_cols))

    for set in parameters:
        Dmat = Dmat + ((U_1 * set[0]) - (U_2 * set[3]))
        Dvec = U_1 * np.transpose(set[1]) + 2 * U_2 * np.matmul(set[3], set[4])

    Dmat = Dmat / len(cov_samples)
    Dvec = Dvec / len(cov_samples)

    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, len(ev_cols), name="Z_DV", lb=-10000, ub=10000)  # DV for decision variable

    # Add objective function
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)

    #This can be improved by including concavity into the decision
    qm.parameters.optimalitytarget.set(3)

    qm.solve()
    qm.print_solution()
