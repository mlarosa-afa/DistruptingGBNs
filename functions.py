import math
import random
import numpy as np
from scipy.stats import invwishart
from docplex.mp.model import Model

#holds the subsets of Lambda
class Lambda:
    def __init__(self, Lambda_dot, evidence_vars, unobserved_vars):
        self.yy = np.delete(Lambda_dot, evidence_vars, 1)
        self.yy = np.delete(self.yy, evidence_vars, 0)
        self.yz = np.delete(Lambda_dot, unobserved_vars, 1)
        self.yz = np.delete(self.yz, evidence_vars, 0)
        self.zy = np.transpose(self.yz)
        self.zz = np.delete(Lambda_dot, unobserved_vars, 1)
        self.zz = np.delete(self.zz, unobserved_vars, 0)

class nonstandardVars:
    def __init__(self, Q, vT, c, K,h,g, K_prime,h_prime,u_prime, S_prime, L):
        self.Q, self.vT, self.c, self.K, self.h, self.g, self.K_prime, self.h_prime, self.u_prime, self.S_prime, self.L = Q, vT, c, K,h,g, K_prime,h_prime,u_prime, S_prime, L

#Generates a Positive Definate square matrix of dimention num_dim.
#There may be room for optimization here
def generate_pos_def_matrix(num_dim, seed = 0):
    if not seed == 0:
        random.seed(seed)

    #runs until matrix is Pos_def
    while True:
        # https://math.stackexchange.com/questions/332456/how-to-make-a-matrix-positive-semidefinite
        random_matrix = np.random.rand(num_dim, num_dim)
        PSD_Random = np.dot(random_matrix, random_matrix.transpose())
        # check to see if PD, if so, continnue
        if is_pos_def(PSD_Random):
            break
    return PSD_Random

#Determins if given matrix X is positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

#Calculate the optimal (max) weights (phi_1 & phi_2)
def  solve_optimal_weights(Q, vT, c, K_prime, u_prime, NUM_EVIDENCE_VARS):
    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, NUM_EVIDENCE_VARS, name="Z_DV", lb=-3, ub=3)  # DV for decision variable

    # Solve max KL first
    Dmat = Q
    Dvec = np.transpose(vT)
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)
    qm.parameters.optimalitytarget.set(2)
    qm.solve()
    bestKL = qm.objective_value

    # Solve marginal mode second
    Dmat = -1 * (K_prime)
    Dvec = 2 * np.matmul(K_prime, u_prime)
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)
    qm.parameters.optimalitytarget.set(2)
    qm.solve()
    bestMM = qm.objective_value

    return bestKL, bestMM

#Select which variables are evidence given MVG prior and Mean
#Optional Parameters:
# Number of evidence variables to select. Will default to 25% of variables
# evidence variables. list of rows. Will overwrite num of evidence vars if available
def generate_evidence(MVG_Sigma, MVG_Mu, evidence_vars=[], NUM_EVIDENCE_VARS = 0, seed=3):
    # If the number of evidence variables is not provided, make it 25% of the variables
    if NUM_EVIDENCE_VARS == 0:
        if not len(evidence_vars) == 0:
            NUM_EVIDENCE_VARS = len(evidence_vars)
        else:
            NUM_EVIDENCE_VARS = math.floor(MVG_Mu.size * .25)


    random.seed(seed)
    # Generate Random evidence if none provided
    if len(evidence_vars) == 0:
        evidence_vars = random.sample(range(MVG_Mu.shape[0]), NUM_EVIDENCE_VARS)
        evidence_vars.sort()

    unobserved_vars = list(set(list(range(MVG_Mu.size))) - set(evidence_vars))

    # Generates a new observed valued centered on MVG_Mu with sd of MVG_Sigma
    np.random.seed(seed)
    observed_vals = list()
    for j in evidence_vars:
        observed_vals.append(np.random.normal(MVG_Mu[j], MVG_Sigma[j, j], 1)[0])

    return evidence_vars, unobserved_vars, observed_vals


def params_from_sample(MVG_Sigma, MVG_Mu, evidence_vars, unobserved_vars, observed_vals):

    # Is this inverse required?
    Lambda_dot = np.linalg.inv(MVG_Sigma)  #Precision matrix of joint evidence distribution
    eta_dot = np.matmul(Lambda_dot, MVG_Mu)
    Xi_dot = -0.5 * np.matmul(np.matmul(np.transpose(MVG_Mu), Lambda_dot), MVG_Mu) - math.log((2*math.pi)**(MVG_Mu.size/2) * (np.linalg.det(MVG_Sigma) ** 0.5))

    # Lambda Subsets  - Form Block Matrices from Joint Covariance Matrix
    L = Lambda(Lambda_dot, evidence_vars, unobserved_vars)

    Sigma_zz = np.delete(MVG_Sigma, unobserved_vars, 1)
    Sigma_zz = np.delete(Sigma_zz, unobserved_vars, 0)

    #eta Subsets
    eta_y = np.delete(eta_dot, evidence_vars)
    eta_z = np.delete(eta_dot, unobserved_vars)

    # Conditional Distribution
    K = L.yy
    h = eta_y - np.matmul(L.yz, observed_vals)
    g = Xi_dot + np.matmul(np.transpose(eta_z), observed_vals) - (0.5 * np.matmul(np.matmul(np.transpose(observed_vals), L.zz), observed_vals))

    u_dot = np.matmul(np.linalg.inv(K), h)

    #KL Divergence
    K_prime = L.zz - np.matmul(np.matmul(L.zy,np.linalg.inv(L.yy)),L.yz)
    h_prime = eta_z - np.matmul(L.zy, np.matmul(np.linalg.inv(L.yy), eta_y))

    S_prime = np.linalg.inv(K_prime)
    u_prime = np.matmul(S_prime, h_prime)

    Q = np.matmul(np.matmul(np.transpose(L.yz), np.linalg.inv(K)), L.yz)
    vT = 2 * (np.matmul(np.transpose(u_dot), L.yz) - np.matmul(np.matmul(np.transpose(eta_y), np.linalg.inv(K)), L.yz))
    c = np.matmul(np.matmul(np.transpose(u_dot), K), u_dot) - 2 * np.matmul(np.transpose(eta_y), u_dot) + np.matmul(np.transpose(eta_y), np.matmul(np.linalg.inv(K), eta_y))
    vals = nonstandardVars(Q, vT, c, K,h,g, K_prime,h_prime,u_prime, S_prime, L)
    return vals

#given optimals and eigenvalues, calculates concavity
def identify_convavity(rho, Phi_1_opt, Zeta, Phi_2_opt, NUM_EVIDENCE_VARS):
    #identify best constraints
    b_concave = np.inf
    b_convex = -np.inf
    for rho_index in range(1, len(rho)+1):
        for zeta_index in range(1, len(Zeta)+1):
            const = ((Zeta[zeta_index-1] / Phi_2_opt) / ((rho[rho_index-1] / Phi_1_opt) + (Zeta[zeta_index-1] / Phi_2_opt)))
            if rho_index + zeta_index - 1 <= NUM_EVIDENCE_VARS:
                if const < b_concave:
                    b_concave = const
            if rho_index + zeta_index - NUM_EVIDENCE_VARS <= NUM_EVIDENCE_VARS and 1 <= rho_index + zeta_index - NUM_EVIDENCE_VARS:
                if const > b_convex:
                    b_convex = const

    print(rho, Zeta)

    return b_concave, b_convex

#Identifys concavity given a prior
#If no prior is passed, a random PD matrix of dim n is created
def whitebox_preprocessing(MVG_Sigma=np.array([]), n=20, NUM_EVIDENCE_VARS = 0, seed = 3):

    #checks to see if prior is passed. If not, generate a PD matrix as prior
    if MVG_Sigma.size == 0:
        MVG_Sigma = generate_pos_def_matrix(n)

    # calculate row mean from cov matrix
    MVG_Mu = np.mean(MVG_Sigma, axis=0)

    evidence_vars, unobserved_vars, observed_vals = generate_evidence(MVG_Sigma, MVG_Mu, NUM_EVIDENCE_VARS = NUM_EVIDENCE_VARS, seed = seed)
    v = params_from_sample(MVG_Sigma, MVG_Mu, evidence_vars, unobserved_vars, observed_vals)

    Phi_opt1, Phi_opt2 = solve_optimal_weights(v.Q, v.vT, v.c, v.K_prime, v.u_prime, len(evidence_vars))

    rho = np.sort(np.linalg.eigvals(v.Q))
    Zeta = np.sort(np.linalg.eigvals(v.L.zz))

    b_concave, b_convex = identify_convavity(rho, Phi_opt1, Zeta, Phi_opt2, len(evidence_vars))
    return b_concave, b_convex, Phi_opt1, Phi_opt2, v, evidence_vars, observed_vals, unobserved_vars

#SDG Methodss
def check_gd_bounds(constraint, curr_pos, step):
    next_step = curr_pos + step
    #if(sum(constraint(curr_pos + step)) == len(curr_pos)):
    #    return curr_pos + step
    if next_step[0] > 3:
        next_step[0] = 3
    if next_step[1] > 3:
        next_step[1] = 3
    if next_step[2] > 3:
        next_step[2] = 3
    if next_step[0] < -3:
        next_step[0] = -3
    if next_step[1] < -3:
        next_step[1] = -3
    if next_step[2] < -3:
        next_step[2] = -3

    return next_step

def adaGrad(gradient, constraint, curr_pos, learn_rate=0.001, v = 0):
    step = (learn_rate / math.sqrt(v + (.00001))) * gradient(curr_pos)
    v_t_1 = v + pow(np.linalg.norm(gradient(curr_pos)), 2)
    return check_gd_bounds(constraint, curr_pos, step), v_t_1

def RMSProp(gradient, constraint, curr_pos, learn_rate= 0.001, v = 0, beta = .9):
    step = ((learn_rate / math.sqrt(v + (.00001))) * gradient(curr_pos))
    v_t_1 = (beta * v) + ((1-beta) * pow(np.linalg.norm(gradient(curr_pos)), 2))
    return check_gd_bounds(constraint, curr_pos, step), v_t_1

def adam(gradient, constraint, curr_pos, learn_rate= 0.001, t = 0, v = 0, m = 0, beta_1 = .9, beta_2 = .999):
    m_t_1 = (beta_1 * m) + ((1 - beta_1) * gradient(curr_pos))
    v_t_1 = (beta_1 * v) + ((1 - beta_2) * pow(np.linalg.norm(gradient(curr_pos)), 2))
    m_t_1_hat = m_t_1 / (1 - pow(beta_1, t))
    v_t_1_hat = v_t_1 / (1 - pow(beta_2, t))
    step = (learn_rate / (np.sqrt(v_t_1_hat) + .00001)) * m_t_1_hat
    return check_gd_bounds(constraint, curr_pos, step), v_t_1, m_t_1
