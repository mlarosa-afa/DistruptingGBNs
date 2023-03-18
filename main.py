import random
from functions import *
from scipy.stats import invwishart, multivariate_normal
import numpy as np

from docplex.mp.model import Model

# Pulling Prior from bnlean's gaussian.test (R Package)
MVG_Sigma = np.array([
    [2.222218, 2.405899, 13.16801, 2.767075, 6.961947],
    [2.405899, 9.518770, 27.97397, 13.044076, 17.126274],
    [13.168007, 27.973973, 106.47261, 36.051377, 58.480076],
    [2.767075, 13.044076, 36.05138, 18.136263, 23.240560],
    [6.961947, 17.126274, 58.48008, 23.240560, 49.670134]])

if __name__ == '__main__':

    print(
        "Mode 1: Whitebox\nMode 2: Gray Box - Sample Average Approximation\nMode 3: Gray Box - Stochastic Gradient Descent\n")
    mode = int(input("Enter which version you would like to run: "))
    if mode == 1:

        U_1 = float(input("Enter U_1: "))
        U_2 = 1 - U_1
        print("Caluclated weight 2 as ", U_2)
        num_evidence = int(input("Number of Evidence Variables: "))

        MVG_Mu = np.mean(MVG_Sigma, axis=0)

        b_concave, b_convex, Phi_opt1, Phi_opt2, v, evidence_vars, observed_vals, unobserved_vars = whitebox_preprocessing(MVG_Sigma,
                                                                                                          NUM_EVIDENCE_VARS=num_evidence, seed = 12)
        print("The interesting range of weight 1 ranges from ", b_concave, " to ", b_convex)

        qm = Model('DistruptionGBN')
        idx = {1: {1: [1] * num_evidence}}
        z_DV = qm.continuous_var_matrix(1, num_evidence, name="Z_DV", lb=-3, ub=3)  # DV for decision variable

        with open("WBTable.tsv", "w+") as f:
            f.write("u_1\tu_2\tz_1\tz_2\tz_3\tObj\tϕ_1\tϕ_2\n")
            iter = np.concatenate((np.arange(0.0, 1.0, 0.01), np.arange(b_concave, b_convex, 0.001)))
            iter.sort()
            for i in iter:
                U_1 = i
                U_2 = 1-i
                # Solve normalized problem
                W_1 = U_1 / Phi_opt1
                W_2 = U_2 / Phi_opt2

                Dmat = (W_1 * v.Q) - (W_2 * v.K_prime)
                Dvec = W_1 * np.transpose(v.vT) + 2 * W_2 * np.matmul(v.K_prime, v.u_prime)
                obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
                qm.set_objective("max", obj_fn)

                qm.parameters.optimalitytarget.set(3)
                #qm.parameters.optimalitytarget.set(2)

                solutionset = qm.solve()
                #qm.print_solution()
                z = np.array([qm.solution["Z_DV_0_0"], qm.solution["Z_DV_0_1"], qm.solution["Z_DV_0_2"]])

                Sigma_zz = np.delete(MVG_Sigma, unobserved_vars, 1)
                Sigma_zz = np.delete(Sigma_zz, unobserved_vars, 0)

                MVG_Mu_z = np.delete(MVG_Mu, unobserved_vars)

                phi_1 = np.transpose(z) @ v.Q @ z + np.transpose(v.vT) @ z
                phi_2 = (-1 * np.transpose(z) @ np.linalg.inv(Sigma_zz) @ z) + (2 * np.transpose(z) @ np.linalg.inv(Sigma_zz)@ MVG_Mu_z)

                f.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (U_1, U_2, qm.solution["Z_DV_0_0"], qm.solution["Z_DV_0_1"], qm.solution["Z_DV_0_2"], qm.objective_value, phi_1, phi_2))

    elif mode == 2:
        numDim = 7  # int(input("Enter number of Dimentions: "))
        numSamples = 1000  # int(input("Enter number of Samples: "))
        numDf = 100  # int(input("Enter degrees of freedom: "))

        W_1 = float(input("Enter U 1: "))
        W_2 = float(input("Enter U 2: "))
        NUM_EVIDENCE_VARS = int(input("Enter Number of Observered Variables: "))

        prior = generate_pos_def_matrix(numDim)

        invWishart = invwishart.rvs(df=numDf, scale=np.identity(len(prior)), size=numSamples)


        # Q, vT, c, K_prime, u_prime
        parameters = []

        for sample_Sigma in invWishart:
            sample_Mu = multivariate_normal.rvs(cov=sample_Sigma)
            evidence_vars, unobserved_vars, observed_vals = generate_evidence(sample_Sigma, sample_Mu, NUM_EVIDENCE_VARS=NUM_EVIDENCE_VARS, seed=3)
            vals = params_from_sample(sample_Sigma, sample_Mu, evidence_vars, unobserved_vars, observed_vals)
            parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))

        Dmat = np.zeros((NUM_EVIDENCE_VARS, NUM_EVIDENCE_VARS))
        Dvec = np.zeros(NUM_EVIDENCE_VARS)

        for set in parameters:
            Dmat = Dmat + ((W_1 * set[0]) - (W_2 * set[3]))
            Dvec = W_1 * np.transpose(set[1]) + 2 * W_2 * np.matmul(set[3], set[4])

        qm = Model('DistruptionGBN')
        # idx = {1: {1: [1] * NUM_EVIDENCE_VARS}}
        z_DV = qm.continuous_var_matrix(1, NUM_EVIDENCE_VARS, name="Z_DV", lb=-10000,
                                        ub=10000)  # DV for decision variable

        # Add objective function
        obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
        qm.set_objective("max", obj_fn)

        qm.parameters.optimalitytarget.set(2)
        qm.solve()
        qm.print_solution()
        print(np.linalg.eig(Dmat))

    elif mode == 3:
        print("running SGD")

        numDim = 5 # int(input("Enter number of Dimentions: "))
        numSamples = 100  # int(input("Enter number of Samples: "))
        numDf = 10  # int(input("Enter degrees of freedom: "))

        W_1 = 0#float(input("Enter U 1: "))
        W_2 = 1 - W_1
        NUM_EVIDENCE_VARS = 3 #float(input("Enter Number of observered Variables: "))

        psi = generate_pos_def_matrix(numDim, seed = 19)

        evidence_vars, unobserved_vars, observed_vals = generate_evidence(psi,  multivariate_normal.rvs(cov=psi),
                                                                          NUM_EVIDENCE_VARS=NUM_EVIDENCE_VARS, seed = 19)

        print("Please select a method:\n\t1.AdaGrad\n\t2.RMSProp\n\t3.Adam")
        method = int(input("method:"))
        LEARN_RATE = float(input("Learning Rate: "))

        position = np.array([0, 0, 0])
        prev_position = np.array([10000, 10000, 10000])
        v = 0
        t = 1
        m = 0

        #number of iters

        while abs(np.linalg.norm(position-prev_position)) > .0001:
            sample_cov = invwishart.rvs(df=numDf, scale=psi)
            sample_Mu = multivariate_normal.rvs(cov=sample_cov)

            vals = params_from_sample(sample_cov, sample_Mu, evidence_vars, unobserved_vars, observed_vals)
            Dmat = ((W_1 * vals.Q) - (W_2 * vals.K_prime))
            Dvec = W_1 * np.transpose(vals.vT) + 2 * W_2 * np.matmul(vals.K_prime, vals.u_prime)
            prev_position = position
            if method == 1:
                position, v = adaGrad(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), lambda z: z <= 3, position, LEARN_RATE, v = v)
            elif method == 2:
                position, v = RMSProp(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), lambda z: z <= 3, position, LEARN_RATE, v=v)
            elif method == 3:
                position, v, m = adam(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), lambda z: z <= 3, position, LEARN_RATE, t = t, v = v, m = m)
                t = t + 1
            print(position)

    print("Thanks for using this program")
