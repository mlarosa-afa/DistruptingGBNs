from docplex.mp.model import Model
import random
import numpy as np
from scipy.stats import invwishart
import math, time


def solveB1(rho, Phi_1_opt, Zeta, Phi_2_opt, NUM_EVIDENCE_VARS):
    mdl = Model(name='B1')

    u1 = mdl.continuous_var(name='u1')
    mdl.add_constraint(u1 >= 0, 'ct1')
    mdl.add_constraint(u1 <= 10000, 'ct2')



    b_concave = np.inf
    b_convex = -np.inf
    for rho_index in range(1,len(rho)+1):
        for zeta_index in range(1,len(Zeta)+1):
            const = ((Zeta[zeta_index-1] / Phi_2_opt) / ((rho[rho_index-1] / Phi_1_opt) + (Zeta[zeta_index-1] / Phi_2_opt)))
            if rho_index + zeta_index - 1 <= NUM_EVIDENCE_VARS:
                if const < b_concave:
                    b_concave = const
            if rho_index + zeta_index - NUM_EVIDENCE_VARS <= NUM_EVIDENCE_VARS and 1 <= rho_index + zeta_index - NUM_EVIDENCE_VARS:
                if const > b_convex:
                    b_convex = const

                #mdl.add_constraint(((rho[rho_index-1]/Phi_1_opt) * u1) - ((Zeta[zeta_index-1]/Phi_2_opt) * (1-u1)) <= 0, 'ct'+ str(i))
                #i = i + 1
    
    #mdl.maximize(u1)
    #mdl.solve()

    return b_concave, b_convex


    """
    for v in mdl.iter_continuous_vars():
        #print(v.solution_value)
        for rho_index in range(1, len(rho) + 1):
            for zeta_index in range(1, len(Zeta) + 1):
                if rho_index + zeta_index - 1 <= NUM_EVIDENCE_VARS:
                    #print(((rho[rho_index - 1] / Phi_1_opt) * v.solution_value) - ((Zeta[zeta_index - 1] / Phi_2_opt) * (1 - v.solution_value)))
                    pass

        return v.solution_value
    """

def  solve_optimal_weights(Q, vT, c, K_prime, u_prime, NUM_EVIDENCE_VARS):
    qm = Model('DistruptionGBN')
    idx = {1: {1: [1] * NUM_EVIDENCE_VARS}}
    z_DV = qm.continuous_var_matrix(1, NUM_EVIDENCE_VARS, name="Z_DV", lb=-3, ub=3)  # DV for decision variable

    Dmat = np.zeros((NUM_EVIDENCE_VARS, NUM_EVIDENCE_VARS))
    Dvec = np.zeros(NUM_EVIDENCE_VARS)

    # Solve max KL first
    Dmat = Q
    Dvec = np.transpose(vT)
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)
    qm.parameters.optimalitytarget.set(2)
    qm.solve()
    # qm.print_solution()
    bestKL = qm.objective_value

    # Solve marginal mode second
    Dmat = -1 * (K_prime)
    Dvec = 2 * np.matmul(K_prime, u_prime)
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)
    qm.parameters.optimalitytarget.set(2)
    qm.solve()
    # qm.print_solution()
    bestMM = qm.objective_value

    return bestKL, bestMM

def vals_from_priors(MVG_Sigma, MVG_Mu, NUM_EVIDENCE_VARS = 1):
    Lambda_dot = np.linalg.inv(MVG_Sigma)  #Precision matrix of joint evidence distribution
    eta_dot = np.matmul(Lambda_dot, MVG_Mu)
    Xi_dot = -0.5 * np.matmul(np.matmul(np.transpose(MVG_Mu), Lambda_dot), MVG_Mu) - math.log((2*math.pi)**(MVG_Mu.size/2) * (np.linalg.det(MVG_Sigma) ** 0.5))

    #Generate Random evidence
    evidence_vars = random.sample(range(MVG_Mu.shape[0]), NUM_EVIDENCE_VARS)
    evidence_vars.sort()

    observed_vals = list()
    for j in evidence_vars:
        observed_vals.append(np.random.normal(MVG_Mu[j], MVG_Sigma[j, j], 1)[0])

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

    Sigma_zz = np.delete(MVG_Sigma, unobserved_vars, 1)
    Sigma_zz = np.delete(Sigma_zz, unobserved_vars, 0)

    #eta Subsets
    eta_y = np.delete(eta_dot, evidence_vars)
    eta_z = np.delete(eta_dot, unobserved_vars)

    # Conditional Distribution
    K = Lambda_yy
    h = eta_y - np.matmul(Lambda_yz, observed_vals)
    g = Xi_dot + np.matmul(np.transpose(eta_z), observed_vals) - (0.5 * np.matmul(np.matmul(np.transpose(observed_vals), Lambda_zz), observed_vals))

    u_dot = np.matmul(np.linalg.inv(K), h)

    #KL Divergence
    K_prime = Lambda_zz - np.matmul(np.matmul(Lambda_zy,np.linalg.inv(Lambda_yy)),Lambda_yz)
    h_prime = eta_z - np.matmul(Lambda_zy, np.matmul(np.linalg.inv(Lambda_yy), eta_y))

    S_prime = np.linalg.inv(K_prime)
    u_prime = np.matmul(S_prime, h_prime)

    Q = np.matmul(np.matmul(np.transpose(Lambda_yz), np.linalg.inv(K)), Lambda_yz)
    vT = 2 * (np.matmul(np.transpose(u_dot), Lambda_yz) - np.matmul(np.matmul(np.transpose(eta_y), np.linalg.inv(K)), Lambda_yz))
    c = np.matmul(np.matmul(np.transpose(u_dot), K), u_dot) - 2 * np.matmul(np.transpose(eta_y), u_dot) + np.matmul(np.transpose(eta_y), np.matmul(np.linalg.inv(K), eta_y))

    Phi_opt1, Phi_opt2 = solve_optimal_weights(Q, vT, c, K_prime, u_prime, NUM_EVIDENCE_VARS)
    return Q, Sigma_zz, Lambda_zz, Phi_opt1, Phi_opt2, NUM_EVIDENCE_VARS

def perform_wb(n = 20):

    while True:
        # create PSD Matarix
        # https://math.stackexchange.com/questions/332456/how-to-make-a-matrix-positive-semidefinite
        random_matrix = np.random.rand(n, n)
        PSD_Random = np.dot(random_matrix, random_matrix.transpose())
        # check to see if PD, if so, continnue
        if is_pos_def(PSD_Random):
            break

    # PSD_random is Gaurenteed PD at this point
    # Using default size=1, random_state=None
    iw = invwishart.rvs(df=len(PSD_Random), scale=PSD_Random)
    iw = np.linalg.inv(iw)
    # calculate row mean from cov matrix
    iw_mean = np.mean(iw, axis=0)
    #math.floor(iw_mean.shape[0] * .25)
    #print(n, PSD_Random)
    return vals_from_priors(iw, iw_mean, math.floor(n * .25))

#https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
#Assuming Symetric (Is that okay?)
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


if __name__ == '__main__':
    random.seed(263)
    times = []
    for i in np.repeat(np.arange(10, 400 + 1, step=5), 20):
        s_t = time.time()

        Q, Sigma_zz, Lambda_zz, Phi_opt1, Phi_opt2, NUM_EV = perform_wb(i)
        rho = np.sort(np.linalg.eigvals(Q))
        Zeta = np.sort(np.linalg.eigvals(Lambda_zz))
    
        b_concave, b_convex = solveB1(rho, Phi_opt1, Zeta, Phi_opt2, NUM_EV)
        #print("Concave: ", b_concave, "Convex: ", b_convex)
        #upper_u = 0.99993 for seed 2023
    
        w1 = b_concave / Phi_opt1
        w2 = (1 - b_concave) / Phi_opt2
        little_lambda_eigs = np.linalg.eigvals((w1 * Q) - (w2 * Lambda_zz))
        
        #print("CONCAVE: ", b_concave, little_lambda_eigs)
    
        w1 = b_convex / Phi_opt1
        w2 = (1 - b_convex) / Phi_opt2
        little_lambda_eigs = np.linalg.eigvals((w1 * Q) - (w2 * Lambda_zz))
    
        #print("CONVEX: ", b_convex, little_lambda_eigs)
        f_t = time.time() - s_t
        times.append((i, f_t))
    """     
    while(1):
        b_convex = float(input("please test"))

        w1 = b_convex / Phi_opt1
        w2 = (1 - b_convex) / Phi_opt2
        little_lambda_eigs = np.linalg.eigvals((w1 * Q) - (w2 * Lambda_zz))
        print("TEST: ", b_convex, little_lambda_eigs)
    """

    print("AVERAGE TIMES")
    curr_time = 0
    tot_time = 0
    tot_elements = 0
    avg_times = []
    for element in times:
        if not (element[0] == curr_time):
            if tot_elements > 0:
                avg_time = tot_time / tot_elements
                avg_times.append((curr_time, avg_time))
            curr_time = element[0]
            tot_time = 0
            tot_elements = 0
        tot_time = tot_time + element[1]
        tot_elements = tot_elements + 1


    for i in avg_times:
        print(i[0], "\t", i[1])



