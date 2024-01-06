from docplex.mp.model import Model
import numpy as np
from functions import *

def gb_SAA(cov_samples, mu_samples, ev_cols, evidence, U_1, U_2, phi_1opt, phi_2opt, ev_bounds = None, risk_tolerance = .1, optimality_target = 3):
    """
        Executes Sample Average Approximation Attack

        Parameter
        ---------
        cov_samples : array
            samples of covariance matrix. Must have same dimension as mu_sample.
        mu_samples : array
            samples of mean. must have same dimensions as cov_samples.
        ev_cols : array
            array specifying the index column/row of evidence variables. Must have same dimension as evidence.
        evidence : array
            array specifying the true observed values of ev_cols. Must have same dimension as ev_cols.
        U_1 : float
            unnormalized weight for KL Divergence. For interpretability, U_1 and U_2 should add to 1.
        U_2 : float
            unnormalized weight for log densities. For interpretability, U_1 and U_2 should add to 1.
        ev_bounds : 2D array
            2xn array where row 1 provides the maximum constraints for each evidence variable and row
            2 provided the minimum constraint for each evidence variable.
            Row 1 MUST be > Row 2 for all entries. If no bounds are provided, +- 10% deviation is assumed.
        risk_tolerance : float
            if ev_bounds is not provided, risk_tolerance modifies the default +- 10% deviation to specified value.
        optimality_target : int
            CPLEX optimality target. See https://www.ibm.com/docs/en/icos/20.1.0?topic=parameters-optimality-target
            for more information

        Returns
        ---------
        obj_value : float
            value of objection function at the proposed evidence
        solution : docplex solution class
            solution of posioned evidence values in "Z_DV_X" field where X is the evidence index
        Phi_1 : float
            solution for phi1 normalized
        Phi_2 : float
            solution for phi2 normalized
    """

    #Compute evidence range
    if ev_bounds is None:
        ev_bounds = np.stack(([x * (1+risk_tolerance) for x in evidence], [x * (1 - risk_tolerance) for x in evidence]))

    # Q, vT, c, K_prime, u_prime
    parameters = []

    #Convert Sample to normal form
    for i in range(len(cov_samples)):
        vals = vals_from_priors(cov_samples[i], mu_samples[i], ev_cols, evidence)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))  #Note uses canonical formto find Sigmazz^-1 and mu[Z}
    average_params = np.array(parameters, dtype=object)
    average_params = average_params.mean(axis=0)

    Aphi_1opt = abs(phi_1opt)
    Aphi_2opt = abs(phi_2opt)

    # Solve normalized problem
    W_1 = U_1 / Aphi_1opt
    W_2 = U_2 / Aphi_2opt

    Dmat = ((W_1 * average_params[0]) - (W_2 * average_params[3]))
    Dvec = W_1 * np.transpose(average_params[1]) + 2 * W_2 * np.matmul(average_params[3], average_params[4])

    obj_value, solution = solveqm(Dmat, Dvec, len(ev_cols), ev_bounds)

    #Gather evidence in 1D array
    evidence = np.array([])
    for i in range(len(ev_cols)):
        evidence = np.append(evidence, solution["Z_DV_" + str(i)])

    #calculate phi1/phi2
    phi_1 = np.transpose(evidence) @ (W_1 * Dmat) @ evidence + np.transpose(evidence) @ Dvec
    phi_2 = np.transpose(evidence) @ (-1 * W_2 * average_params[3]) @ evidence + (
                np.transpose(evidence) @ (2 * W_2 * np.matmul(average_params[3], average_params[4])))
    phi_1 = phi_1 / Aphi_1opt
    phi_2 = phi_2 / Aphi_2opt

    return obj_value, solution, phi_1, phi_2



