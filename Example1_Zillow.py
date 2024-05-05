import random

import numpy as np
from functions import *
from wb_attack import whitebox_attack
from baseline_analysis import evaluate_objective, random_attack
from gb_SAA import gb_SAA, saa_phi_opt_est
from gb_SGD import gb_SGD
import time
import sys

from tabulate import tabulate

"""
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



MVG_Sigma = np.array([[5633.13720182397,3953.56350442939,5696.54502995966,5908.40846865913,3132.65281424638,4685.95853971231,7125.40107655088,2482.53883188861,4870.54324221806,3045.39425505274,4394.53426225957,3094.64361100225,2464.75536133702,3137.88399450192,970.932982264266],
[3953.56350442939,2783.16988231891,3999.28102685402,4143.48339608434,2206.38383860305,3289.19209435244,5009.20400733079,1744.84604912695,3425.66237358499,2145.39931540289,3089.93463840296,2177.24822478076,1732.91079520448,2211.76658833468,689.762417592559],
[5696.54502995966,3999.28102685402,5775.10645755913,5975.25883389975,3167.3792232314,4738.95930736773,7214.7311080428,2502.76251211361,4931.81659470485,3089.92996491742,4448.05628612968,3127.64332800765,2483.5551705973,3175.27807073293,981.144822255607],
[5908.40846865913,4143.48339608434,5975.25883389975,6218.66982371205,3279.57345697831,4931.85014189848,7458.74435047344,2609.56770668009,5106.95528880397,3183.52726417422,4611.52242277671,3247.995918704,2592.92037074519,3287.20383135781,1005.44594094879],
[3132.65281424638,2206.38383860305,3167.3792232314,3279.57345697831,1755.00541370846,2602.03355085721,3976.28598751154,1383.43108212308,2713.89515160189,1701.20071809942,2450.0136440433,1724.89953656941,1369.51894975935,1754.02044209754,543.784447421731],
[4685.95853971231,3289.19209435244,4738.95930736773,4931.85014189848,2602.03355085721,3915.5701230095,5910.86717641299,2072.24173332203,4054.5266961397,2524.32886177112,3659.38538991646,2580.7647365343,2062.67560153114,2611.55507144056,804.710857726163],
[7125.40107655088,5009.20400733079,7214.7311080428,7458.74435047344,3976.28598751154,5910.86717641299,9059.74480845138,3129.21430852162,6170.118733385,3880.45939690134,5564.25234169895,3910.12181963438,3097.02795511457,3974.69915800556,1232.38645708968],
[2482.53883188861,1744.84604912695,2502.76251211361,2609.56770668009,1383.43108212308,2072.24173332203,3129.21430852162,1108.60623855288,2145.79823856301,1332.19737991904,1940.40454979227,1369.55460889158,1102.175906388,1388.35780997718,434.157813784582],
[4870.54324221806,3425.66237358499,4931.81659470485,5106.95528880397,2713.89515160189,4054.5266961397,6170.118733385,2145.79823856301,4223.90733302859,2644.42450763464,3806.42084810969,2682.33867657319,2133.23657180538,2721.2777112293,852.577112890332],
[3045.39425505274,2145.39931540289,3089.92996491742,3183.52726417422,1701.20071809942,2524.32886177112,3880.45939690134,1332.19737991904,2644.42450763464,1672.07074638476,2380.36783693424,1672.42629038916,1320.37314942014,1703.26775027765,533.457598906657],
[4394.53426225957,3089.93463840296,4448.05628612968,4611.52242277671,2450.0136440433,3659.38538991646,5564.25234169895,1940.40454979227,3806.42084810969,2380.36783693424,3435.89292083302,2418.90246588142,1925.80318296575,2455.42478186063,762.385154009007],
[3094.64361100225,2177.24822478076,3127.64332800765,3247.995918704,1724.89953656941,2580.7647365343,3910.12181963438,1369.55460889158,2682.33867657319,1672.42629038916,2418.90246588142,1708.71862578358,1363.94927150554,1730.73124041286,543.195590857245],
[2464.75536133702,1732.91079520448,2483.5551705973,2592.92037074519,1369.51894975935,2062.67560153114,3097.02795511457,1102.175906388,2133.23657180538,1320.37314942014,1925.80318296575,1363.94927150554,1102.6689720086,1379.57339274853,438.156984896257],
[3137.88399450192,2211.76658833468,3175.27807073293,3287.20383135781,1754.02044209754,2611.55507144056,3974.69915800556,1388.35780997718,2721.2777112293,1703.26775027765,2455.42478186063,1730.73124041286,1379.57339274853,1761.1956830554,552.448584820142],
[970.932982264266,689.762417592559,981.144822255607,1005.44594094879,543.784447421731,804.710857726163,1232.38645708968,434.157813784582,852.577112890332,533.457598906657,762.385154009007,543.195590857245,438.156984896257,552.448584820142,207.220820180105]])
MVG_mu = np.array([332.823814591464,243.985668404878,261.218564897561,363.589741642683,182.822203369512,244.082363564634,429.786776240244,167.151774896341,239.382750287805,182.17597452561,237.685315231707,177.700859819512,179.807453914634,184.921322840244,89.2414639643902])
"""
MVG_Sigma = np.array([[0.563313720182397, 0.39535635, 0.569654503, 0.590840847, 0.313265281, 0.468595854, 0.712540108,
                       0.248253883, 0.487054324, 0.304539426, 0.439453426, 0.309464361, 0.246475536, 0.313788399,
                       0.0970932982264266],
                      [0.395356350442939, 0.278316988, 0.399928103, 0.41434834, 0.220638384, 0.328919209, 0.500920401,
                       0.174484605, 0.342566237, 0.214539932, 0.308993464, 0.217724822, 0.17329108, 0.221176659,
                       0.0689762417592559],
                      [0.569654502995966, 0.399928103, 0.577510646, 0.597525883, 0.316737922, 0.473895931, 0.721473111,
                       0.250276251, 0.493181659, 0.308992996, 0.444805629, 0.312764333, 0.248355517, 0.317527807,
                       0.0981144822255607],
                      [0.590840846865913, 0.41434834, 0.597525883, 0.621866982, 0.327957346, 0.493185014, 0.745874435,
                       0.260956771, 0.510695529, 0.318352726, 0.461152242, 0.324799592, 0.259292037, 0.328720383,
                       0.100544594094879],
                      [0.313265281424638, 0.220638384, 0.316737922, 0.327957346, 0.175500541, 0.260203355, 0.397628599,
                       0.138343108, 0.271389515, 0.170120072, 0.245001364, 0.172489954, 0.136951895, 0.175402044,
                       0.0543784447421731],
                      [0.468595853971231, 0.328919209, 0.473895931, 0.493185014, 0.260203355, 0.391557012, 0.591086718,
                       0.207224173, 0.40545267, 0.252432886, 0.365938539, 0.258076474, 0.20626756, 0.261155507,
                       0.0804710857726163],
                      [0.712540107655088, 0.500920401, 0.721473111, 0.745874435, 0.397628599, 0.591086718, 0.905974481,
                       0.312921431, 0.617011873, 0.38804594, 0.556425234, 0.391012182, 0.309702796, 0.397469916,
                       0.123238645708968],
                      [0.248253883188861, 0.174484605, 0.250276251, 0.260956771, 0.138343108, 0.207224173, 0.312921431,
                       0.110860624, 0.214579824, 0.133219738, 0.194040455, 0.136955461, 0.110217591, 0.138835781,
                       0.0434157813784582],
                      [0.487054324221806, 0.342566237, 0.493181659, 0.510695529, 0.271389515, 0.40545267, 0.617011873,
                       0.214579824, 0.422390733, 0.264442451, 0.380642085, 0.268233868, 0.213323657, 0.272127771,
                       0.0852577112890332],
                      [0.304539425505274, 0.214539932, 0.308992996, 0.318352726, 0.170120072, 0.252432886, 0.38804594,
                       0.133219738, 0.264442451, 0.167207075, 0.238036784, 0.167242629, 0.132037315, 0.170326775,
                       0.0533457598906657],
                      [0.439453426225957, 0.308993464, 0.444805629, 0.461152242, 0.245001364, 0.365938539, 0.556425234,
                       0.194040455, 0.380642085, 0.238036784, 0.343589292, 0.241890247, 0.192580318, 0.245542478,
                       0.0762385154009007],
                      [0.309464361100225, 0.217724822, 0.312764333, 0.324799592, 0.172489954, 0.258076474, 0.391012182,
                       0.136955461, 0.268233868, 0.167242629, 0.241890247, 0.170871863, 0.136394927, 0.173073124,
                       0.0543195590857245],
                      [0.246475536133703, 0.17329108, 0.248355517, 0.259292037, 0.136951895, 0.20626756, 0.309702796,
                       0.110217591, 0.213323657, 0.132037315, 0.192580318, 0.136394927, 0.110266897, 0.137957339,
                       0.0438156984896258],
                      [0.313788399450192, 0.221176659, 0.317527807, 0.328720383, 0.175402044, 0.261155507, 0.397469916,
                       0.138835781, 0.272127771, 0.170326775, 0.245542478, 0.173073124, 0.137957339, 0.176119568,
                       0.0552448584820142],
                      [0.0970932982264266, 0.068976242, 0.098114482, 0.100544594, 0.054378445, 0.080471086, 0.123238646,
                       0.043415781, 0.085257711, 0.05334576, 0.076238515, 0.054319559, 0.043815698, 0.055244858,
                       0.0207220820180105]])
MVG_mu = np.array([3.328238146,
                   2.439856684,
                   2.612185649,
                   3.635897416,
                   1.828222034,
                   2.440823636,
                   4.297867762,
                   1.671517749,
                   2.393827503,
                   1.821759745,
                   2.376853152,
                   1.777008598,
                   1.798074539,
                   1.849213228,
                   0.89241464])

ev_vars = [0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 14]
evidence = [1.45885, 1.21453, 1.34413, 1.37732, .94179, 1.53126, .83859, .76678, .67944, .85680, .53759]
ev_bounds = np.stack(([x + .15 for x in evidence], [x - .15 for x in evidence]))

def zillow_baseline(U_1, concavityFlag=False, timeFlag=False, verbose=True):
    start_time = time.time()
    obj_value, phi_1, phi_2 = evaluate_objective(MVG_Sigma, MVG_mu, ev_vars, evidence, evidence, U_1, ev_bounds=ev_bounds)
    end_time = time.time()
    
    if verbose:
        print(f"Solution: Zillow Baseline Analysis\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}")
        print("\nEvidence Analytics")
        print(tabulate([["Objective Value", obj_value]]))
        print(f"\nPoint Estimates of Unobserved Means")
        point_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, evidence)
        print(tabulate([["Y_"+str(i), point_estimates[i]] for i in range(len(point_estimates))]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds=ev_bounds)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
            
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))
    
    return obj_value, phi_1, phi_2

def zillow_random(U_1, concavityFlag=False, timeFlag=False, verbose=True, seed=2023):
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    obj_val, solution, phi_1, phi_2 = random_attack(MVG_Sigma, MVG_mu, ev_vars, evidence, U_1, ev_bounds=ev_bounds)
    end_time = time.time()

    #print(f"Objective Value with random attack: {obj_value} \n Utilized random evidence:{proposed_evidence}")
    if verbose:
        print(f"Solution: Zillow Random Analysis\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}")
        print("\nPoisioned Attack")
        print(tabulate([["Z_"+str(i), round(solution[i],2)] for i in range(len(solution))], headers=['Evidence', 'Posioned Value']))
        print("\nAttack Analytics")
        print(tabulate([["Objective Value", obj_val],["Phi_1",phi_1],["Phi_2",phi_2]]))
        print(f"\nKL Divergence from True Evidence")
        print(tabulate([["KL", KL_divergence(MVG_Sigma, MVG_mu, ev_vars, evidence, solution)]]))
        print(f"\nPoint Estimates of Unobserved Means")
        point_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, solution)
        print(tabulate([["Y_"+str(i), point_estimates[i]] for i in range(len(point_estimates))]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds=ev_bounds)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
            
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))
    
    return obj_val, solution, phi_1, phi_2

def zillow_wb(U_1, concavityFlag=False, timeFlag=False, verbose=True):

    start_time = time.time()
    obj_val, solution, phi_1, phi_2 = whitebox_attack(MVG_Sigma, MVG_mu, ev_vars, evidence, U_1, ev_bounds=ev_bounds)
    end_time = time.time()

    if verbose:
        print(f"Solution: Zillow Whitebox Attack\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}")
        print("\nPoisioned Attack")
        print(tabulate([["Z_"+str(i), round(solution[i],2)] for i in range(len(solution))], headers=['Evidence', 'Posioned Value']))
        print("\nAttack Analytics")
        print(tabulate([["Objective Value", obj_val],["Phi_1",phi_1],["Phi_2",phi_2]]))
        print(f"\nKL Divergence from True Evidence")
        print(tabulate([["KL", KL_divergence(MVG_Sigma, MVG_mu, ev_vars, evidence, solution)]]))
        print(f"\nPoint Estimates of Unobserved Means")
        point_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, solution)
        print(tabulate([["Y_"+str(i), point_estimates[i]] for i in range(len(point_estimates))]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds=ev_bounds)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
        
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))

    return obj_val, solution, phi_1, phi_2

phi_1opt, phi_2opt = -241747.29447119022, 32906.527971408235
def zillow_saa(U_1, numSamples, PsiMultiplier, mu_notMultiplier, KAPPA, nu, concavityFlag=False, timeFlag=False, verbose=True, seed=14):
    np.random.seed(seed)
    random.seed(seed)

    Psi = (PsiMultiplier * MVG_Sigma)
    mu_not = mu_notMultiplier * MVG_mu

    #phi_1opt, phi_2opt = saa_phi_opt_est(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds, PsiMultiplier, mu_notMultiplier, KAPPA, nu)

    start_time = time.time()
    cov_samples = invwishart.rvs(df=nu, scale=Psi, size=numSamples)

    mu_samples = []
    for cov_sample in cov_samples:
        mu_samples.append(np.random.multivariate_normal(mu_not, (1 / KAPPA) * cov_sample))

    obj_val, solution, phi_1, phi_2 = gb_SAA(cov_samples, mu_samples, ev_vars, evidence, U_1/abs(phi_1opt), (1-U_1)/abs(phi_2opt), ev_bounds=ev_bounds)
    end_time = time.time()

    if verbose:
        print(f"Solution: Zillow Grey Box (SAA) Attack\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}, Ψ = {PsiMultiplier}, μ_0 = {mu_notMultiplier}, κ = {KAPPA}, ν = {nu}")
        print("\nPoisioned Attack")
        print(tabulate([["Z_"+str(i), round(solution[i],2)] for i in range(len(solution))], headers=['Evidence', 'Posioned Value']))
        print("\nAttack Analytics")
        print(tabulate([["Objective Value", obj_val],["Phi_1",phi_1],["Phi_2",phi_2]]))
        print(f"\nKL Divergence from True Evidence")
        print(tabulate([["KL", KL_divergence(MVG_Sigma, MVG_mu, ev_vars, evidence, solution)]]))
        print(f"\nPoint Estimates of Unobserved Means")
        point_estimates = point_estimates_means(MVG_Sigma, MVG_mu, ev_vars, solution)
        print(tabulate([["Y_"+str(i), point_estimates[i]] for i in range(len(point_estimates))]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds=ev_bounds)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
        
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))

    return obj_val, solution, phi_1, phi_2


def zillow_sgd(U_1, PsiMultiplier, mu_notMultiplier, KAPPA, nu, method, LEARN_RATE, epsilon, concavityFlag=False, timeFlag=False, verbose=True):
    Psi = PsiMultiplier * MVG_Sigma
    mu_not = mu_notMultiplier * MVG_mu

    #phi_1opt, phi_2opt = saa_phi_opt_est(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds, PsiMultiplier, mu_notMultiplier, KAPPA, nu)

    start_time = time.time()
    warm_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prev_solution = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
                              float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')])
    obj_val, solution, phi_1, phi_2 = gb_SGD(warm_start, prev_solution, Psi, mu_not, ev_vars, evidence, method,
                               U_1 / abs(phi_1opt), (1-U_1) / abs(phi_2opt), ev_bounds,
                               LEARN_RATE=LEARN_RATE, nu=nu, KAPPA=KAPPA, epsilon=epsilon)
    end_time = time.time()

    if verbose:
        methods = {1:"AdaGrad", 2:"RMSProp", 3:"Adam"}
        print(f"Solution: Zillow Grey Box (SGA) Attack\n   Parameterized under U_1 = {U_1}, U_2 = {1-U_1}, Ψ = {PsiMultiplier}, μ_0 = {mu_notMultiplier}, κ = {KAPPA}, ν = {nu}")
        print(f"\tmethod = {methods[method]}, lr = {LEARN_RATE}, ε  = {epsilon}")
        print("\nPoisioned Attack")
        print(tabulate([["Z_"+str(i), round(solution[i],2)] for i in range(len(solution))], headers=['Evidence', 'Posioned Value']))
        print("\nAttack Analytics")
        print(tabulate([["Objective Value", obj_val],["Phi_1",phi_1],["Phi_2",phi_2]]))
        print(f"\nKL Divergence from True Evidence")
        print(tabulate([["KL", KL_divergence(MVG_Sigma, MVG_mu, ev_vars, evidence, solution)]]))

        if concavityFlag == True:
            print("\nConcavity Analytics")
            b_concave, b_convex = convavity_from_conanical(MVG_Sigma, MVG_mu, ev_vars, evidence, ev_bounds=ev_bounds)
            print(tabulate([["U_1-", b_concave],["U_1+",b_convex]]))
        
        if timeFlag == True:
            print("\nTime Analytics")
            print(tabulate([["Time Elapsed (s)", end_time - start_time]]))
    
    return obj_val, solution, phi_1, phi_2