from cplex import Model
import numpy as np
from functions import identify_convavity, vals_from_priors, solve_optimal_weights


def whitebox_attack(MVG_Sigma, MVG_Mu, evidence_vars, evidence, U_1, U_2, optimality_target=3):

    v = vals_from_priors(MVG_Sigma, MVG_Mu, evidence_vars, evidence)

    Phi_opt1, Phi_opt2 = solve_optimal_weights(v.Q, v.vT, v.K_prime, v.u_prime, len(evidence_vars))

    rho = np.sort(np.linalg.eigvals(v.Q))
    Zeta = np.sort(np.linalg.eigvals(v.L.zz))

    b_concave, b_convex = identify_convavity(rho, Phi_opt1, Zeta, Phi_opt2, len(evidence_vars))

    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, len(evidence_vars), name="Z_DV", lb=-3, ub=3)  # DV for decision variable

    # Solve normalized problem
    W_1 = U_1 / Phi_opt1
    W_2 = U_2 / Phi_opt2

    Dmat = (W_1 * v.Q) - (W_2 * v.K_prime)
    Dvec = W_1 * np.transpose(v.vT) + 2 * W_2 * np.matmul(v.K_prime, v.u_prime)
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)

    qm.parameters.optimalitytarget.set(optimality_target)

    solutionset = qm.solve()
    qm.print_solution()

    return b_concave, b_convex, solutionset, qm
