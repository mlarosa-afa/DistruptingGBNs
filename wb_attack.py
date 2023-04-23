import numpy as np
from functions import identify_convavity, vals_from_priors, solve_optimal_weights, solveqm


def whitebox_attack(MVG_Sigma, MVG_mu, evidence_vars, evidence, u_1, u_2, ev_bounds=None, risk_tolerance=.1,
                    optimality_target=3):
    # Compute evidence range
    if ev_bounds is None:
        ev_bounds = np.stack(([x * (1+risk_tolerance) for x in evidence], [x * (1-risk_tolerance) for x in evidence]))

    v = vals_from_priors(MVG_Sigma, MVG_mu, evidence_vars, evidence)

    phi_opt1, phi_opt2 = solve_optimal_weights(v.Q, v.vT, v.K_prime, v.u_prime, len(evidence_vars), ev_bounds=ev_bounds)

    rho = np.sort(np.real(np.linalg.eigvals(v.Q)))[::-1]
    #Zeta = np.sort(np.real(np.linalg.eigvals(v.L.zz)))[::-1]
    invSigma_zz= np.linalg.inv(v.Sigma_zz)
    Zeta = np.sort(np.real(np.linalg.eigvals(invSigma_zz)))[::-1]

    b_concave, b_convex = identify_convavity(rho, phi_opt1, Zeta, phi_opt2, len(evidence_vars))
    Aphi_opt1 = abs(phi_opt1)
    Aphi_opt2 = abs(phi_opt2)
    # Solve normalized problem
    w_1 = u_1 / Aphi_opt1
    w_2 = u_2 / Aphi_opt2

    Dmat = (w_1 * v.Q) - (w_2 * v.K_prime)
    Dvec = w_1 * np.transpose(v.vT) + 2 * w_2 * np.matmul(v.K_prime, v.u_prime)

    obj_value, solution_set = solveqm(Dmat, Dvec, len(evidence_vars), ev_bounds=ev_bounds, optimality_target=optimality_target)

    #Temp code here
    evidence = np.array([solution_set["Z_DV_0"],solution_set["Z_DV_1"],solution_set["Z_DV_2"],solution_set["Z_DV_3"],solution_set["Z_DV_4"],solution_set["Z_DV_5"],solution_set["Z_DV_6"],solution_set["Z_DV_7"],solution_set["Z_DV_8"],solution_set["Z_DV_9"],solution_set["Z_DV_10"]])
    phi_1 = np.transpose(evidence)@(w_1 * v.Q)@evidence + np.transpose(evidence) @ v.vT
    phi_2 = np.transpose(evidence)@(-1* w_2 * v.K_prime) @ evidence + (np.transpose(evidence) @ (2 * w_2 * np.matmul(v.K_prime, v.u_prime)))
    phi_1 = phi_1/Aphi_opt1
    phi_2 = phi_2 / Aphi_opt2
    #temp code here

    return b_concave, b_convex, solution_set, obj_value, phi_1, phi_2
