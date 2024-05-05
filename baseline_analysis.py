import random
import numpy as np
from functions import identify_convavity, vals_from_priors, solve_optimal_weights
 
def evaluate_objective(MVG_Sigma, MVG_mu, ev_cols, true_evidence, proposed_evidence, U_1, ev_bounds=None, risk_tolerance=.1):
    """
        Provides the objective value of the true evidence
 
        Parameter
        ---------
        MVG_Sigma : array
            Covariance matrix of model
        MVG_Mu : array
            Mean of model
        ev_cols : array
            array specifying the index column/row of evidence variables. Must have same dimension as evidence.
        true_evidence : array
            array specifying the true observed values of ev_cols. Must have same dimension as ev_cols. This is z' in the paper.
        proposed_evidence : array
            array specifying the attack of ev_cols. Must have same dimension as ev_cols. This is z in the paper.
        U_1 : float
            unnormalized weight for KL Divergence. For interpretability, U_1 and U_2 should add to 1.
        ev_bounds : 2D array
            2xn array where row 1 provides the maximum constraints for each evidence variable and row
            2 provided the minimum constraint for each evidence variable.
            Row 1 MUST be > Row 2 for all entries. If no bounds are provided, +- 10% deviation is assumed.
        risk_tolerance : float
            if ev_bounds is not provided, risk_tolerance modifies the default +- 10% deviation to specified value.

        Returns
        ---------
        obj_value : float
            value of objection function at the true evidence
        Phi_1 : float
            solution for phi1 normalized
        Phi_2 : float
            solution for phi2 normalized
    """
 
    # Compute evidence range
    if ev_bounds is None:
        ev_bounds = np.stack(([x * (1+risk_tolerance) for x in true_evidence], [x * (1-risk_tolerance) for x in true_evidence]))
 
    #Change Forms
    v = vals_from_priors(MVG_Sigma, MVG_mu, ev_cols, true_evidence)
 
    #Calculate Normalized Weights
    phi_opt1, phi_opt2 = solve_optimal_weights(v.Q, v.vT, v.K_prime, v.u_prime, len(ev_cols), ev_bounds=ev_bounds)
    W_1 = U_1 / abs(phi_opt1)
    W_2 = (1-U_1) / abs(phi_opt2)
 
    #Sovle Phi_1 and Phi_2
    phi_1 = np.transpose(proposed_evidence)@(W_1 * v.Q)@proposed_evidence + np.transpose(proposed_evidence) @ (W_1*v.vT)
    phi_2 = np.transpose(proposed_evidence)@(-1 * W_2 * v.K_prime) @ proposed_evidence + (np.transpose(proposed_evidence) @ (2 * W_2 * np.matmul(v.K_prime, v.u_prime)))

    return phi_1+phi_2, phi_1, phi_2
 
 
 
def random_attack(MVG_Sigma, MVG_mu, ev_cols, true_evidence, U_1, ev_bounds=None, risk_tolerance=.1):
    """
        Executes random attack within the evidence threshold
 
        Parameter
        ---------
        MVG_Sigma : array
            Covariance matrix of model
        MVG_Mu : array
            Mean of model
        ev_cols : array
            array specifying the index column/row of evidence variables. Must have same dimension as evidence.
        true_evidence : array
            array specifying the true observed values of ev_cols. Must have same dimension as ev_cols.
        U_1 : float
            unnormalized weight for KL Divergence. For interpretability, U_1 and U_2 should add to 1.
        ev_bounds : 2D array
            2xn array where row 1 provides the maximum constraints for each evidence variable and row
            2 provided the minimum constraint for each evidence variable.
            Row 1 MUST be &gt; Row 2 for all entries. If no bounds are provided, +- 10% deviation is assumed.
        risk_tolerance : float
            if ev_bounds is not provided, risk_tolerance modifies the default +- 10% deviation to specified value.
 
 
        Returns
        ---------
        obj_value : float
            value of objection function at randomly sampled evidence
        evidence : array
            randomly sampled (proposed) evidence
        Phi_1 : float
            solution for phi1 normalized
        Phi_2 : float
            solution for phi2 normalized
    """
 
    # Compute evidence range
    if ev_bounds is None:
        ev_bounds = np.stack(([x * (1+risk_tolerance) for x in true_evidence], [x * (1-risk_tolerance) for x in true_evidence]))

    #Select random proposal
    proposed_evidence = np.array([])
    for i in range(len(ev_cols)):
        proposed_evidence = np.append(proposed_evidence, random.uniform(ev_bounds[0][i], ev_bounds[1][i]))

    #Evaluate at proposed evidence
    obj_val, phi_1, phi_2 = evaluate_objective(MVG_Sigma, MVG_mu,ev_cols, true_evidence, proposed_evidence, U_1, ev_bounds)
    
    return obj_val, proposed_evidence, phi_1, phi_2
