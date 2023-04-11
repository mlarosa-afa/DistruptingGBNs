import numpy as np
from functions import *

from wb_attack import whitebox_attack
from gb_SAA import gb_SAA
from gb_SGD import gb_SGD

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

MVG_mu = np.array([332823.81,243985.67,261218.56,363589.74,182822.20,244082.36,429786.78,167151.77,239382.75,182175.97,237685.32,177700.86,179807.45,184921.32,89241.46])

mode = int(input("1) Whitebox Attack\n2) Graybox - Sample Average Approximation\n3) Graybox - Stocastic Gradient Descent\nSelected Mode: "))
if mode == 1:
    U_1 = float(input("Enter U_1: "))
    U_2 = 1 - U_1
    print("Caluclated weight 2 as ", U_2)
    ev_vars = [0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 14]
    evidence = [145885, 121453, 134413, 137732, 94179, 153126, 83859, 76678, 67944, 85680, 53759]

    b_concave, b_convex, solutionset, obj_value = whitebox_attack(MVG_Sigma, MVG_mu, ev_vars, evidence, U_1, U_2)
    print("The interesting range of weight 1 ranges from ", b_concave, " to ", b_convex)
    print(solutionset)

elif mode == 2:
    Psi = MVG_Sigma
    mu_not = MVG_mu
    KAPPA = 4
    nu = len(mu_not)

    U_1 = float(input("Enter U 1: "))
    U_2 = 1 - U_1
    print("Caluclated weight 2 as ", U_2)
    numSamples = int(input("Enter number of Samples: "))
    ev_vars = [0,1,2,3,5,6,8,10,11,12,14]
    evidence = [145885, 121453, 134413, 137732, 94179, 153126, 83859, 76678, 67944, 85680, 53759]
    upper_bounds = [x + 15000 for x in evidence]
    lower_bounds = [x - 15000 for x in evidence]
    ev_bounds = np.stack((upper_bounds, lower_bounds))

    cov_samples = invwishart.rvs(df=nu, scale=Psi, size=numSamples)

    mu_samples = []
    for cov_sample in cov_samples:
        mu_samples.append(np.random.multivariate_normal(mu_not, (1/KAPPA) * cov_sample))

    solution = gb_SAA(cov_samples, mu_samples, ev_vars, evidence, U_1, U_2, ev_bounds=ev_bounds)
    print(solution)

elif mode == 3:
    U_1 = float(input("Enter U 1: "))
    U_2 = 1 - U_1
    print("Caluclated weight 2 as ", U_2)
    nu = int(input("Enter degrees of freedom: "))

    ev_vars = [0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 14]
    evidence = [145885, 121453, 134413, 137732, 94179, 153126, 83859, 76678, 67944, 85680, 53759]

    print("Please select a method:\n\t1.AdaGrad\n\t2.RMSProp\n\t3.Adam")
    method = int(input("method: "))
    LEARN_RATE = float(input("Learning Rate: "))

    solution = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    prev_solution = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')])
    ev_bounds = np.stack(([x + 15000 for x in evidence], [x - 15000 for x in evidence]))

    phi_opt1, solution = gb_SGD(solution, prev_solution, MVG_Sigma, MVG_mu, ev_vars, evidence, method, 1, 0, ev_bounds, LEARN_RATE=LEARN_RATE, nu=nu)

    solution = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    prev_solution = np.array(
        [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
         float('inf'), float('inf'), float('inf')])
    phi_opt2, warm_start = gb_SGD(solution, prev_solution, MVG_Sigma, MVG_mu, ev_vars, evidence, method, 0, 1, ev_bounds, LEARN_RATE=LEARN_RATE, nu=nu)

    prev_solution = np.array(
        [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
         float('inf'), float('inf'), float('inf')])
    obj_val, solution = gb_SGD(warm_start, prev_solution, MVG_Sigma, MVG_mu, ev_vars, evidence, method, U_1 / phi_opt1, U_2 / phi_opt2, ev_bounds, LEARN_RATE=LEARN_RATE, nu=nu)
    print("Objection Function Value: ", obj_val, " at ", solution)