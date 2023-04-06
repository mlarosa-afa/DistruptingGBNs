import numpy as np
from functions import *
from scipy.stats import invwishart, multivariate_normal

MVG_Sigma = np.array([[5633137202, 3953563504, 5696545030, 5908408469, 3132652814, 4685958540, 7125401077, 2482538832, 4870543242, 3045394255, 4394534262, 3094643611, 2464755361, 3137883995, 970932982],
[3953563504, 2783169882, 3999281027, 4143483396, 2206383839, 3289192094, 5009204007, 1744846049, 3425662374, 2145399315, 3089934638, 2177248225, 1732910795, 2211766588, 689762418],
[5696545030, 3999281027, 5775106458, 5975258834, 3167379223, 4738959307, 7214731108, 2502762512, 4931816595, 3089929965, 4448056286, 3127643328, 2483555171, 3175278071, 981144822],
[5908408469, 4143483396, 5975258834, 6218669824, 3279573457, 4931850142, 7458744350, 2609567707, 5106955289, 3183527264, 4611522423, 3247995919, 2592920371, 3287203831, 1005445941],
[3132652814, 2206383839, 3167379223, 3279573457, 1755005414, 2602033551, 3976285988, 1383431082, 2713895152, 1701200718, 2450013644, 1724899537, 1369518950, 1754020442, 543784447],
[4685958540, 3289192094, 4738959307, 4931850142, 2602033551, 3915570123, 5910867176, 2072241733, 4054526696, 2524328862, 3659385390, 2580764737, 2062675602, 2611555071, 804710858],
[7125401077, 5009204007, 7214731108, 7458744350, 3976285988, 5910867176, 9059744808, 3129214309, 6170118733, 3880459397, 5564252342, 3910121820, 3097027955, 3974699158, 1232386457],
[2482538832, 1744846049, 2502762512, 2609567707, 1383431082, 2072241733, 3129214309, 1108606239, 2145798239, 1332197380, 1940404550, 1369554609, 1102175906, 1388357810, 434157814],
[4870543242, 3425662374, 4931816595, 5106955289, 2713895152, 4054526696, 6170118733, 2145798239, 4223907333, 2644424508, 3806420848, 2682338677, 2133236572, 2721277711, 852577113],
[3045394255, 2145399315, 3089929965, 3183527264, 1701200718, 2524328862, 3880459397, 1332197380, 2644424508, 1672070746, 2380367837, 1672426290, 1320373149, 1703267750, 533457599],
[4394534262, 3089934638, 4448056286, 4611522423, 2450013644, 3659385390, 5564252342, 1940404550, 3806420848, 2380367837, 3435892921, 2418902466, 1925803183, 2455424782, 762385154],
[3094643611, 2177248225, 3127643328, 3247995919, 1724899537, 2580764737, 3910121820, 1369554609, 2682338677, 1672426290, 2418902466, 1708718626, 1363949272, 1730731240, 543195591],
[2464755361, 1732910795, 2483555171, 2592920371, 1369518950, 2062675602, 3097027955, 1102175906, 2133236572, 1320373149, 1925803183, 1363949272, 1102668972, 1379573393, 438156985],
[3137883995, 2211766588, 3175278071, 3287203831, 1754020442, 2611555071, 3974699158, 1388357810, 2721277711, 1703267750, 2455424782, 1730731240, 1379573393, 1761195683, 552448585],
[970932982, 689762418, 981144822, 1005445941, 543784447, 804710858, 1232386457, 434157814, 852577113, 533457599, 762385154, 543195591, 438156985, 552448585, 207220820]])

MVG_mu = np.array([332823.81,243985.67,261218.56,363589.74,182822.20,244082.36,244082.36,429786.78,167151.77,239382.75,182175.97,237685.32,177700.86,179807.45,184921.32,89241.46])

mode = int(input("1) Whitebox Attack\n2) Graybox - Sample Average Approximation\n3) Graybox - Stocastic Gradient Descent\nSelected Mode: "))
if mode == 1:
    U_1 = float(input("Enter U_1: "))
    U_2 = 1 - U_1
    print("Caluclated weight 2 as ", U_2)
    num_evidence = int(input("Number of Evidence Variables: "))

    b_concave, b_convex, Phi_opt1, Phi_opt2, v, evidence_vars, observed_vals, unobserved_vars = whitebox_preprocessing(
        MVG_Sigma,
        NUM_EVIDENCE_VARS=num_evidence, seed=12)
    print("The interesting range of weight 1 ranges from ", b_concave, " to ", b_convex)

    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, num_evidence, name="Z_DV", lb=-3, ub=3)  # DV for decision variable

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
    """
    #Create a list of all solution vars
    sol_param_names = []
    for i in range(0, num_evidence):
        sol_param_names.append("Z_DV_0_" + str(i))
    
    z = []
    for var_name in sol_param_names:
        z.append(qm.solution[var_name])
    z = np.array(z)

    Sigma_zz = np.delete(MVG_Sigma, unobserved_vars, 1)
    Sigma_zz = np.delete(Sigma_zz, unobserved_vars, 0)

    MVG_Mu_z = np.delete(MVG_mu, unobserved_vars)

    phi_1 = np.transpose(z) @ v.Q @ z + np.transpose(v.vT) @ z
    phi_2 = (-1 * np.transpose(z) @ np.linalg.inv(Sigma_zz) @ z) + (
                2 * np.transpose(z) @ np.linalg.inv(Sigma_zz) @ MVG_Mu_z)

    print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
    U_1, U_2, qm.solution["Z_DV_0_0"], qm.solution["Z_DV_0_1"], qm.solution["Z_DV_0_2"], qm.objective_value,
    phi_1, phi_2))
    """

elif mode == 2:
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