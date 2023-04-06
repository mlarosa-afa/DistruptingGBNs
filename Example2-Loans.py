import numpy as np
from functions import *
from scipy.stats import invwishart, multivariate_normal

MVG_Sigma = np.array([[4.185882e+09, -175182.27926,1410540839.1,563470612.6,6.280136e+09,   2.172400e+08,-8190.29377],
                      [-1.751823e+05,225.18981,    191876.1,    26066.5,    2.118087e+05,   8.711134e+03,10.52829],
                      [1.410541e+09, 191876.10111, 2882210596.6,414822294.7,5.245630e+09,   1.215825e+08,29660.39985],
                      [5.634706e+08, 26066.49798,  414822294.7, 706308549.2,2.062676e+09,   8.763860e+07,52309.89728],
                      [6.280136e+09, 211808.72361, 5245630110.1,2062675879.5,3.525107e+10,  5.879936e+08,26872.26282],
                      [2.172400e+08, 8711.13438,   121582519.9, 87638601.0, 5.879936e+08,   1.061313e+08,7505.96752],
                      [-8.190294e+03,10.52829,     29660.4,     52309.9,    2.687226e+04,   7.505968e+03,83.81531]])

mode = int(input("1) Graybox - Sample Average Approximation\n2) Graybox - Stocastic Gradient Descent\nSelected Mode: "))
if mode == 1:
    numSamples = int(input("Enter number of Samples: "))
    numDf = int(input("Enter degrees of freedom: "))

    W_1 = float(input("Enter U 1: "))
    W_2 = 1 - W_1
    NUM_EVIDENCE_VARS = int(input("Enter Number of Observered Variables: "))

    invWishart = invwishart.rvs(df=numDf, scale=MVG_Sigma, size=numSamples)

    # Q, vT, c, K_prime, u_prime
    parameters = []

    for sample_Sigma in invWishart:
        sample_Mu = multivariate_normal.rvs(cov=sample_Sigma)
        evidence_vars, unobserved_vars, observed_vals = generate_evidence(sample_Sigma, sample_Mu,
                                                                          NUM_EVIDENCE_VARS=NUM_EVIDENCE_VARS, seed=3)
        vals = params_from_sample(sample_Sigma, sample_Mu, evidence_vars, unobserved_vars, observed_vals)
        parameters.append((vals.Q, vals.vT, vals.c, vals.K_prime, vals.u_prime))

    Dmat = np.zeros((NUM_EVIDENCE_VARS, NUM_EVIDENCE_VARS))
    Dvec = np.zeros(NUM_EVIDENCE_VARS)

    for set in parameters:
        Dmat = Dmat + ((W_1 * set[0]) - (W_2 * set[3]))
        Dvec = W_1 * np.transpose(set[1]) + 2 * W_2 * np.matmul(set[3], set[4])

    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, NUM_EVIDENCE_VARS, name="Z_DV", lb=-10000, ub=10000)  # DV for decision variable

    # Add objective function
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)

    qm.parameters.optimalitytarget.set(2)
    qm.solve()
    qm.print_solution()
elif mode == 3:

    numDf = int(input("Enter degrees of freedom: "))

    W_1 = float(input("Enter U 1: "))
    W_2 = 1 - W_1
    NUM_EVIDENCE_VARS = int(input("Enter Number of observered Variables: "))

    evidence_vars, unobserved_vars, observed_vals = generate_evidence(MVG_Sigma,  multivariate_normal.rvs(cov=MVG_Sigma), NUM_EVIDENCE_VARS=NUM_EVIDENCE_VARS, seed = 19)

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
        sample_cov = invwishart.rvs(df=numDf, scale=MVG_Sigma)
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
