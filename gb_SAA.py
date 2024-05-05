from docplex.mp.model import Model
import numpy as np
from functions import *

def gb_SAA(cov_samples, mu_samples, ev_cols, evidence, W_1, W_2, ev_bounds = None, risk_tolerance = .1, optimality_target = 3):
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
        W_1 : float
            Normalized weight for KL Divergence. For interpretability, U_1 and U_2 should add to 1.
        W_2 : float
            Normalized weight for KL Divergence. For interpretability, U_1 and U_2 should add to 1.
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
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))  #Note uses canonical form to find Sigmazz^-1 and mu[Z}
    average_params = np.array(parameters, dtype=object)
    average_params = average_params.mean(axis=0)

    #Calculate mat and vec for optimzation
    Dmat = ((W_1 * average_params[0]) - (W_2 * average_params[3]))
    Dvec = W_1 * np.transpose(average_params[1]) + 2 * W_2 * np.matmul(average_params[3], average_params[4])

    obj_value, solution = solveqm(Dmat, Dvec, len(ev_cols), ev_bounds, optimality_target=optimality_target)

    #Gather evidence in 1D array
    proposed_evidence = np.array([])
    for i in range(len(ev_cols)):
        proposed_evidence = np.append(proposed_evidence, solution["Z_DV_" + str(i)])

    #calculate phi1. We can obtain phi_2 = obj_val - phi_1
    phi_1 = np.transpose(proposed_evidence) @ (W_1 * average_params[0]) @ proposed_evidence + np.transpose(proposed_evidence) @ (W_1*average_params[1])

    return obj_value, proposed_evidence, phi_1, obj_value-phi_1


def saa_phi_opt_est(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds, PsiMultiplier, mu_notMultiplier, KAPPA, nu, J=100000, risk_tolerance = 0.1):

    #Compute evidence range
    if ev_bounds is None:
        ev_bounds = np.stack(([x * (1+risk_tolerance) for x in evidence], [x * (1 - risk_tolerance) for x in evidence]))
    
    Psi = PsiMultiplier * MVG_Sigma
    mu_not = mu_notMultiplier * MVG_mu

    #Sample
    cov_samples = invwishart.rvs(df=nu, scale=Psi, size=J)
    mu_samples = []
    for cov_sample in cov_samples:
        mu_samples.append(np.random.multivariate_normal(mu_not, (1 / KAPPA) * cov_sample))

    #Convert forms
    # Q, vT, c, K_prime, u_prime
    parameters = []
    # Convert Sample to normal form
    for i in range(len(cov_samples)):
        vals = vals_from_priors(cov_samples[i], mu_samples[i], ev_vars, evidence)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))  # Note uses canonical formto find Sigmazz^-1 and mu[Z}
    average_params = np.array(parameters, dtype=object)
    average_params = average_params.mean(axis=0)

    #Find optimal Phi
    phi_opt1, phi_opt2 = solve_optimal_weights(average_params[0], average_params[1], average_params[3],
                                               average_params[4], len(ev_vars), ev_bounds=ev_bounds)

    return phi_opt1, phi_opt2

