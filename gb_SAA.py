from docplex.mp.model import Model
import numpy as np
from functions import *

def gb_SAA(cov_samples, mu_samples, ev_cols, evidence, U_1, U_2, ev_bounds = None, risk_tolerance = .1, optimality_target = 3, seed = 12):

    #Compute evidence range
    if ev_bounds is None:
        ev_bounds = np.stack(([x * (1+risk_tolerance) for x in evidence], [x * (1 - risk_tolerance) for x in evidence]))

    # Q, vT, c, K_prime, u_prime
    parameters = []

    #Convert Sample to normal form
    for i in range(len(cov_samples)):
        vals = vals_from_priors(cov_samples[i], mu_samples[i], ev_cols, evidence)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))

    #Solve for Phi_1* and Phi_2*
    Dmat_phi_1 = np.zeros((len(ev_cols), len(ev_cols)))
    Dmat_phi_2 = np.zeros((len(ev_cols), len(ev_cols)))
    Dvec_phi_1 = np.zeros(len(ev_cols))
    Dvec_phi_2 = np.zeros(len(ev_cols))
    for params in parameters:
        Dmat_phi_1 = Dmat_phi_1 + ((1 * params[0]))
        Dvec_phi_1 = Dvec_phi_1 + 1 * np.transpose(params[1])
        Dmat_phi_2 = Dmat_phi_2 - (1 * params[3])
        Dvec_phi_2 = Dvec_phi_2 + 2 * 1 * np.matmul(params[3], params[4])

    Dmat_phi_1 = Dmat_phi_1 / len(cov_samples)
    Dvec_phi_1 = Dvec_phi_1 / len(cov_samples)
    Dmat_phi_2 = Dmat_phi_2 / len(cov_samples)
    Dvec_phi_2 = Dvec_phi_2 / len(cov_samples)

    Phi_opt1, solution = solveqm(Dmat_phi_1, Dvec_phi_1, len(ev_cols), ev_bounds)
    Phi_opt2, solution = solveqm(Dmat_phi_2, Dvec_phi_2, len(ev_cols), ev_bounds)

    # Solve normalized problem
    W_1 = U_1 / Phi_opt1
    W_2 = U_2 / Phi_opt2

    Dmat = np.zeros((len(ev_cols), len(ev_cols)))
    Dvec = np.zeros(len(ev_cols))
    for params in parameters:
        Dmat = Dmat + ((W_1 * params[0]) - (W_2 * params[3]))
        Dvec = Dvec + W_1 * np.transpose(params[1]) + 2 * W_2 * np.matmul(params[3], params[4])

    Dmat = Dmat / len(cov_samples)
    Dvec = Dvec / len(cov_samples)
    obj_value, solution = solveqm(Dmat, Dvec, len(ev_cols), ev_bounds)
    return solution

