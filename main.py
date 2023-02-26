import math, random
from scipy.stats import invwishart, multivariate_normal
import cplex
import numpy as np
import numpy.random

from docplex.mp.model import Model

def vals_from_priors(MVG_Sigma, MVG_Mu):
    Lambda_dot = np.linalg.inv(MVG_Sigma)  #Precision matrix of joint evidence distribution
    eta_dot = np.matmul(Lambda_dot, MVG_Mu)
    Xi_dot = -0.5 * np.matmul(np.matmul(np.transpose(MVG_Mu), Lambda_dot), MVG_Mu) - math.log((2*math.pi)**(MVG_Mu.size/2) * (np.linalg.det(MVG_Sigma) ** 0.5))

    random.seed(100)
    """
    #True Random values
    evidence_vars = random.sample(range(MVG_Mu.size), NUM_EVIDENCE_VARS).sort()

    observed_val = list()
    for j in range(evidence_vars):
        observed_val.append(np.random.normal(MVG_Mu[j], MVG_Sigma[j, j], 1))
    """

    # Values from R
    evidence_vars = [0, 1]
    #observed_vals = [-0.3877692, 125.8079697]
    observed_vals = [0, 0]

    unobserved_vars = list(set(list(range(MVG_Mu.size))) - set(evidence_vars))
    # Lambda Subsets  - Form Block Matrices from Joint Covariance Matrix
    Lambda_yy = np.delete(Lambda_dot, evidence_vars, 1)
    Lambda_yy = np.delete(Lambda_yy, evidence_vars, 0)
    #Lambda_yz = Lambda_dot[!(vars % in% evidence_vars), (vars % in% evidence_vars)]
    Lambda_yz = np.delete(Lambda_dot, unobserved_vars, 1)
    Lambda_yz = np.delete(Lambda_yz, evidence_vars, 0)
    Lambda_zy = np.transpose(Lambda_yz)
    Lambda_zz = np.delete(Lambda_dot, unobserved_vars, 1)
    Lambda_zz = np.delete(Lambda_zz, unobserved_vars, 0)

    #eta Subsets
    eta_y = np.delete(eta_dot, evidence_vars)
    eta_z = np.delete(eta_dot, unobserved_vars)

    # Conditional Distribution
    K = Lambda_yy
    h = eta_y - np.matmul(Lambda_yz, observed_vals)
    g = Xi_dot + np.matmul(np.transpose(eta_z),observed_vals) - (0.5 * np.matmul(np.matmul(np.transpose(observed_vals), Lambda_zz), observed_vals))

    u_dot = np.matmul(np.linalg.inv(K), h)

    #KL Divergence
    K_prime = Lambda_zz - np.matmul(np.matmul(Lambda_zy,np.linalg.inv(Lambda_yy)),Lambda_yz)
    h_prime = eta_z - np.matmul(Lambda_zy, np.matmul(np.linalg.inv(Lambda_yy), eta_y))

    S_prime = np.linalg.inv(K_prime)
    u_prime = np.matmul(S_prime, h_prime)

    Q = np.matmul(np.matmul(np.transpose(Lambda_yz), np.linalg.inv(K)), Lambda_yz)
    vT = 2 * (np.matmul(np.transpose(u_dot), Lambda_yz) - np.matmul(np.matmul(np.transpose(eta_y), np.linalg.inv(K)), Lambda_yz))
    c = np.matmul(np.matmul(np.transpose(u_dot), K), u_dot) - 2 * np.matmul(np.transpose(eta_y), u_dot) + np.matmul(np.transpose(eta_y), np.matmul(np.linalg.inv(K), eta_y))
    return Q, vT, c, K_prime, u_prime

#Pulling Prior from bnlean's gaussian.test (R Package)
MVG_Sigma = np.array([
              [2.222218,  2.405899,   13.16801,   2.767075,   6.961947,   -5.349946,  -16.24868],
              [2.405899,  9.518770,   27.97397,   13.044076,  17.126274,  5.038906,   -21.37693],
              [13.168007, 27.973973,  106.47261,  36.051377,  58.480076,  -16.357502, -108.62405],
              [2.767075,  13.044076,  36.05138,   18.136263,  23.240560,  10.837644,  -24.44388],
              [6.961947,  17.126274,  58.48008,   23.240560,  49.670134,  21.735843,  -41.97427],
              [-5.349946, 5.038906,   -16.35750,  10.837644,  21.735843,  85.079790,  71.66548],
              [-16.248684, -21.376927, -108.62405, -24.443880, -41.974271, 71.665482,  149.08423]])

#MVG_Sig1ma =    np.array([
#             [2,    2,   13,    2,   6,   -5,  -16],
#             [2,     9,   27,   13,  17,  5,   -21],
#             [13,   27,  106,  36,  58,  -16, -108],
#             [2,    13,  36,   18,  23,  10,  -24],
#             [6,    17,  58,   23,  49,  21,  -41],
#             [-5,   5,   -16,  10,  21,  85,  71],
#             [-16, -21, -108, -24, -41, 71,  149]])


#MVG_Mu = np.array([-0.6486452095, -0.0004317675, -0.4559603790,  0.4691271083, -0.6564162950,  2.5464207005,  2.2379555670])
#MVG_Mu = np.array( [5,5,5,5,5,5,5])
MVG_Mu = np.array( [0,5,5,5,5,5,5])

if __name__ == '__main__':
    NUM_EVIDENCE_VARS = 2


    print("Mode 1: Whitebox\nMode 2: Gray Box - Sample Average Approximation\nMode 3: Gray Box - Stochastic Gradient Descent\n")
    mode = int(input("Enter which version you would like to run: "))
    if mode == 1:

        U_1 = float(input("Enter Weight 1: "))
        U_2 = float(input("Enter Weight 2: "))

        Q, vT, c, K_prime, u_prime = vals_from_priors(MVG_Sigma, MVG_Mu)

        qm = Model('DistruptionGBN')
        idx = {1: {1:[1] * NUM_EVIDENCE_VARS}}
        z_DV = qm.continuous_var_matrix(1,NUM_EVIDENCE_VARS, name="Z_DV", lb=-3, ub=3) #DV for decision variable

        Dmat = np.zeros((NUM_EVIDENCE_VARS, NUM_EVIDENCE_VARS))
        Dvec = np.zeros(NUM_EVIDENCE_VARS)

        #Solve max KL first
        Dmat = Q
        Dvec = np.transpose(vT)
        obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
        qm.set_objective("max", obj_fn)
        qm.parameters.optimalitytarget.set(2)
        qm.solve()
        #qm.print_solution()
        bestKL = qm.objective_value

        #Solve marginal mode second
        Dmat =  -1* ( K_prime)
        Dvec = 2 * np.matmul(K_prime, u_prime)
        obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()) )
        qm.set_objective("max", obj_fn)
        qm.parameters.optimalitytarget.set(2)
        qm.solve()
        #qm.print_solution()
        bestMM = qm.objective_value

        #Solve normalized problem
        W_1 = U_1 / bestKL
        W_2 = U_2/ bestMM
        Dmat = (W_1 * Q) - (W_2 * K_prime)
        Dvec = W_1 * np.transpose(vT) + 2 * W_2 * np.matmul(K_prime, u_prime)
        obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()) )
        qm.set_objective("max", obj_fn)
        qm.parameters.optimalitytarget.set(3)
        #qm.parameters.optimalitytarget.set(2)
        qm.solve()
        qm.print_solution()


    elif mode == 2:
        numDim = 7# int(input("Enter number of Dimentions: "))
        numSamples = 1000#int(input("Enter number of Samples: "))
        numDf = 100 #int(input("Enter degrees of freedom: "))

        W_1 = float(input("Enter Weight 1: "))
        W_2 = float(input("Enter Weight 2: "))

        PSD = []
        for i in range(numDim):
            temp = []
            for j in range(numDim):
                temp.append((random.random() + 0.1) * 100)
            PSD.append(temp)

        invWishart = invwishart.rvs(df=numDf, scale=np.identity(len(PSD)), size=numSamples)

        #Q, vT, c, K_prime, u_prime
        parameters = []

        for sample_Sigma in invWishart:
            sample_Mu = multivariate_normal.rvs(cov = sample_Sigma)
            Q, vT, c, K_prime, u_prime = vals_from_priors(sample_Sigma, sample_Mu)
            parameters.append((Q, vT, c, K_prime, u_prime))

        Dmat = np.zeros((NUM_EVIDENCE_VARS, NUM_EVIDENCE_VARS))
        Dvec = np.zeros(NUM_EVIDENCE_VARS)

        for set in parameters:
            Dmat = Dmat + ((W_1 * set[0]) - (W_2 * set[3]))
            Dvec = W_1 * np.transpose(set[1]) + 2 * W_2 * np.matmul(set[3], set[4])

        qm = Model('DistruptionGBN')
        #idx = {1: {1: [1] * NUM_EVIDENCE_VARS}}
        z_DV = qm.continuous_var_matrix(1, NUM_EVIDENCE_VARS, name="Z_DV", lb=-10000, ub=10000)  # DV for decision variable

        # For Testing
        #Dmat = np.identity(3)
        # Dmat[1,2] = 1

        # Add objective function

        obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()) )
        qm.set_objective("max", obj_fn)

        qm.parameters.optimalitytarget.set(2)
        qm.solve()
        qm.print_solution()
        print(np.linalg.eig(Dmat))
    elif mode == 3:
        print("running SGD")



    print("Thanks for using this program")