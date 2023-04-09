from cplex import Model
import numpy as np
from functions import whitebox_preprocessing

def whitebox_attack(MVG_Sigma, MVG_Mu, evidence_vars, U_1, U_2, seed=12):
    b_concave, b_convex, Phi_opt1, Phi_opt2, v = whitebox_preprocessing(
        MVG_Sigma, NUM_EVIDENCE_VARS=len(evidence_vars), seed=seed)

    print("The interesting range of weight 1 ranges from ", b_concave, " to ", b_convex)

    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, len(evidence_vars), name="Z_DV", lb=-3, ub=3)  # DV for decision variable

    # Solve normalized problem
    W_1 = U_1 / Phi_opt1
    W_2 = U_2 / Phi_opt2

    Dmat = (W_1 * v.Q) - (W_2 * v.K_prime)
    Dvec = W_1 * np.transpose(v.vT) + 2 * W_2 * np.matmul(v.K_prime, v.u_prime)
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)

    qm.parameters.optimalitytarget.set(3)
    # qm.parameters.optimalitytarget.set(2)

    solutionset = qm.solve()
    qm.print_solution()
