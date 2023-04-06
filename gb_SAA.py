from cplex import Model
import numpy as np
from scipy.stats import invwishart, multivariate_normal
from functions import *

def gb_SAA(sample_cov, sample_mean, U_1, U_2, seed=12):
    unobserved_vars = list(set(list(range(Psi.size))) - set(evidence_vars))

    # Q, vT, c, K_prime, u_prime
    parameters = []

    for sample_Sigma in sample_cov:


        evidence_vars = generate_evidence(sample_Sigma, sample_Mu, NUM_EVIDENCE_VARS=len(evidence_vars), seed=3)

        vals = params_from_sample(sample_Sigma, sample_Mu, evidence_vars, unobserved_vars, observed_vals)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))

    Dmat = np.zeros((len(evidence_vars), len(evidence_vars)))
    Dvec = np.zeros(len(evidence_vars))

    for set in parameters:
        Dmat = Dmat + ((U_1 * set[0]) - (U_2 * set[3]))
        Dvec = U_1 * np.transpose(set[1]) + 2 * U_2 * np.matmul(set[3], set[4])

    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, len(evidence_vars), name="Z_DV", lb=-10000, ub=10000)  # DV for decision variable

    # Add objective function
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)

    qm.parameters.optimalitytarget.set(2)
    qm.solve()
    qm.print_solution()
