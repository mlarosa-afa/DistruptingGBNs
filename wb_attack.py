import numpy as np
from functions import identify_convavity, vals_from_priors, solve_optimal_weights, solveqm
from baseline_analysis import evaluate_objective

def whitebox_evaluation(v, W_1, W_2, solution):
    #Solve for Phi_1 which allows to backwards solve for Phi_2 sicne obj is known at this point
    phi_1 = np.transpose(solution)@(W_1 * v.Q)@solution + np.transpose(solution) @ (W_1*v.vT)
    phi_2 = np.transpose(solution)@(-1*W_2 * v.K_prime)@solution + (np.transpose(solution) @ (2*W_2*(v.K_prime@v.u_prime)))

    return phi_1 + phi_2, phi_1, phi_2

def whitebox_attack(MVG_Sigma, MVG_mu, ev_cols, true_evidence, U_1, ev_bounds=None, risk_tolerance=.1,
                    optimality_target=3):
    """
        Executes whitebox attack

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
        solution : array
            solution of posioned evidence values in index of ev_cols
        Phi_1 : float
            solution for phi1 normalized
        Phi_2 : float
            solution for phi2 normalized
    """

    # Compute evidence range
    if ev_bounds is None:
        ev_bounds = np.stack(([x * (1+risk_tolerance) for x in true_evidence], [x * (1-risk_tolerance) for x in true_evidence]))

    v = vals_from_priors(MVG_Sigma, MVG_mu, ev_cols, true_evidence)

    #Calculate Normalized Weights
    phi_opt1, phi_opt2 = solve_optimal_weights(v.Q, v.vT, v.K_prime, v.u_prime, len(ev_cols), ev_bounds=ev_bounds)
    W_1 = U_1 / abs(phi_opt1)
    W_2 = (1-U_1) / abs(phi_opt2)

    #Find optimal attack
    Dmat = (W_1 * v.Q) - (W_2 * v.K_prime)
    Dvec = W_1 * np.transpose(v.vT) + 2 * W_2 * np.matmul(v.K_prime, v.u_prime)
    obj_value, solution = solveqm(Dmat, Dvec, len(ev_cols), ev_bounds=ev_bounds, optimality_target=optimality_target)

    #change objective
    proposed_evidence = np.array([])
    for i in range(len(ev_cols)):
        proposed_evidence = np.append(proposed_evidence, solution["Z_DV_" + str(i)])

    _, phi_1, phi_2 = whitebox_evaluation(v, W_1, W_2, proposed_evidence)

    return obj_value, proposed_evidence, phi_1, phi_2